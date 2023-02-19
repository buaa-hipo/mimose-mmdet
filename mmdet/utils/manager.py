import torch, math, time, random
import numpy as np
from torch.autograd import Variable
import torch.utils.checkpoint as checkpoint


def format_size(tensor_size):
    units = ['B', 'KB', 'MB', 'GB']
    for i in range(len(units) - 1):
        if tensor_size <= 1024:
            return f"{tensor_size:.2f} {units[i]}"
        tensor_size /= 1024
    return f"{tensor_size:.2f} {units[-1]}"


def store_rng_state():
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state()
    np_rng_state = np.random.get_state()
    rd_rng_state = random.getstate()
    
    return {
        "torch_rng_state": torch_rng_state,
        "torch_cuda_rng_state": torch_cuda_rng_state,
        "np_rng_state": np_rng_state,
        "rd_rng_state": rd_rng_state,
    }


def restore_rng_state(torch_rng_state=None, torch_cuda_rng_state=None, np_rng_state=None, rd_rng_state=None):
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    if torch_cuda_rng_state is not None:
        torch.cuda.set_rng_state(torch_cuda_rng_state)
    if np_rng_state is not None:
        np.random.set_state(np_rng_state)
    if rd_rng_state is not None:
        random.setstate(rd_rng_state)
        


def cast_checkpoint(func, *args, **kwargs):
    # casted_args = []
    # for arg in args:
    #     if isinstance(arg, int):
    #         casted_args.append(Variable(torch.Tensor([arg])))
    #     else:
    #         casted_args.append(arg)
    # casted_kwargs = {}
    # for k, v in kwargs.items():
    #     if isinstance(arg, int):
    #         casted_kwargs[k] = Variable(torch.Tensor([v]))
    #     else:
    #         casted_kwargs[k] = v
    # return checkpoint.checkpoint(func, *casted_args, **casted_kwargs)
    
    def create_custom_forward(module, **kwargs):
        def custom_forward(*inputs):
            return module(*inputs, **kwargs)
        return custom_forward

    return checkpoint.checkpoint(create_custom_forward(func, **kwargs), *args)



def cast_forward(module_m, name, manager):
    old_forward = module_m.forward
    manager.register_module(name, module_m)
    if len(list(module_m.children())) == 0 or module_m.parameters().__next__().requires_grad == False:
        manager.set_non_checkpoint(name)

    def need_checkpoint():
        return manager.need_checkpoint(name)
    
    def profile_function(func, *args, **kwargs):
        prev_time, prev_allocated = time.time(), torch.cuda.memory_allocated()
        ret = func(*args, **kwargs)
        data = time.time() - prev_time, torch.cuda.memory_allocated() - prev_allocated
        return DataStorage(*data), ret


    def forward(*args, **kwargs):
        """ 共有四种情况 """
        """
            1. 当前层使用 checkpoint
            2. 上层使用 checkpoint
            3. 下层使用 checkpoint
            4. 下层未使用 checkpoint
        """
        if need_checkpoint():
            if manager.is_warmup():
                rng_state = store_rng_state()
                """ warmup 阶段只会checkpoint max_levels // 2的block"""
                t1 = time.time()
                data, ret = profile_function(old_forward, *args, **kwargs)
                t2 = time.time()
                manager.collector_overhead.append(t2 - t1)               
                # checkpoint 会保留输出的 tensor
                # data.mem_allocated[0] -= int(np.array(ret[0].shape).prod() * ret[0].element_size())
                # manager.checkpoint_reduce_memory += data.mem_allocated[0]
                # manager.add_data(name, data)
                restore_rng_state(**rng_state)

                ret = None
                manager.prev_use_checkpoint()
                prev_memory = torch.cuda.memory_allocated()
                
                ret = cast_checkpoint(old_forward, *args, **kwargs)

                data.mem_allocated[0] -= torch.cuda.memory_allocated() - prev_memory
                manager.checkpoint_reduce_memory += data.mem_allocated[0]
                manager.add_data(name, data)
                manager.post_use_checkpoint()
            else:
                manager.prev_use_checkpoint()
                ret = cast_checkpoint(old_forward, *args, **kwargs)
                manager.post_use_checkpoint()
            return ret
        elif manager.under_checkpoint or name in manager.non_checkpoint:
            return old_forward(*args, **kwargs)
        else:
            if manager.is_warmup():
                checkpoint_times = manager.checkpoint_count
                data, ret = profile_function(old_forward, *args, **kwargs)
                if manager.checkpoint_count == checkpoint_times:
                    manager.add_data(name, data)

                return ret
            else:
                return old_forward(*args, **kwargs)
    module_m.forward = forward
    module_m.old_forward = old_forward

    for i, child in enumerate(module_m.children()):
        cast_forward(child, name + "-" + str(i), manager)

def recover_forward(module_m):
    module_m.forward = module_m.old_forward
    for child in module_m.children():
        recover_forward(child)


class DataStorage:
    """ Memory consumption and forward time for a specific input size """
    def __init__(self, time_use, mem_allocated):
        self.time_use = [time_use]
        self.mem_allocated = [mem_allocated]
    
    def add(self, data_storage):
        self.time_use += data_storage.time_use
        self.mem_allocated += data_storage.mem_allocated
    
    def get_time(self):
        return np.mean(self.time_use)
    
    def get_memory(self):
        return np.mean(self.mem_allocated)
    
    def serialize(self):
        return {"time": self.time_use, "memory": self.mem_allocated}
    
    def __str__(self):
        return f"time: {self.get_time() * 1e3:.3f} ms, memory: {format_size(self.get_memory())}"


class PredictFuncObject:
    def __init__(self) -> None:
        pass
    
    def __call__(self):
        raise NotImplementedError("You must implement \"__call__\" method")
    
    def update(self):
        raise NotImplementedError("You must implement \"update\" method")

class PolyPrediction(PredictFuncObject):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.poly_func = self.fit_poly(x, y)

    
    def check_fit(self, poly_func, x, y):
        y_val = poly_func(x)
        if isinstance(y, np.ndarray):
            y_np = y
        else:
            y_np = np.array(y)
        # at least 1 KB
        rel_error =  np.max(np.abs(y_np - y_val) / np.stack([y_np, [1024] * y_np.size]).max(axis=0))
        # error < 10 MB
        fit_flag = rel_error < 0.1 or np.max(np.abs(y_np - y_val)) < 1e7
        if not fit_flag:
            print(f"rel error: {rel_error:.2%}, {np.max(np.abs(y_np - y_val)) / (1024 ** 2):0.2f} MB")
        return fit_flag
    
    def fit_poly(self, x, y, deg=2):
        poly_param = np.polyfit(x, y, deg)
        poly_func = np.poly1d(poly_param)
        if not self.check_fit(poly_func, x, y):
            print(f"Memory consumption cannot be fitted to a quadratic polynomial")
        return poly_func

    def predict(self, *args, **kwargs):
        return self.poly_func(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    def update(self, x, y, update_p=0.2, max_x=1600*1600*3, min_x=400*400*3):
        random_number = math.ceil(1 / update_p)
        random_number = max(random_number, 4)
        random_x = np.array(list(range(random_number))) * (max_x - min_x) / random_number + min_x
        random_y = self.predict(random_x)

        new_x = random_x.tolist()
        new_x.append(x)
        new_y = random_y.tolist()
        new_y.append(y)

        self.poly_func = self.fit_poly(new_x, new_y)


class BaseStrategy:
    def __init__(self) -> None:
        pass

    def get_checkpoint_module(self, *args, **kwargs):
        raise NotImplementedError("You must implement \"get_checkpoint_module\" method")


class GreedyStrategy(BaseStrategy):
    def __init__(self, module_func, non_checkpoint, gc_layers=None) -> None:
        super().__init__()
        self.memory_predict_func = module_func
        if gc_layers is None:
            self.levels_module = self._init_levels_module()
        else:
            self.levels_module = {"0": gc_layers}
        self.non_checkpoint = non_checkpoint
    
    def _init_levels_module(self):
        """ {"0": ["0-1", "0-2", "0-1-1"], "0-1-1": [...]} """
        levels_module = {}
        for name in self.memory_predict_func.keys():
            level = name.count('-')
            if level not in levels_module.keys():
                levels_module[level] = []
            levels_module[level].append(name)

        levels = list(levels_module.keys())
        levels.sort()
        
        def find_direct_parent(name, new_levels_module):
            parent = "0"
            for node in new_levels_module.keys():
                if name.startswith(node):
                    if node.count('-') > parent.count('-') and node.count('-') < name.count('-'):
                        parent = node
            return parent

        new_levels_module = {"0": []}
        for level in levels:
            for name in levels_module[level]:
                parent = find_direct_parent(name, new_levels_module)
                new_levels_module[parent].append(name)
                new_levels_module[name] = []
        return new_levels_module
    

    def sort_by_name(self, bucket):
        # TODO:可以在第一个iter里面设置开始结束时间，以此来确定前后顺序
        # 这里的顺序是后面的module在前面，优先级从低到高
        tag_bucket = []
        for value in bucket:
            layer_index = int(value[-1].split('-')[-1])
            tag_bucket.append((layer_index, value))
        tag_bucket.sort(reverse=True)
        return [value for _, value in tag_bucket]
    
    def split_bucket(self, module_memory: list((float, str))):
        # 显存占用排序，从大到小
        module_memory.sort(reverse=True)
        new_module_memory = []
        i = 0
        while i < len(module_memory):
            # 设置分桶的分界点
            memory_threshold = module_memory[i][0] * 0.9
            bucket = [module_memory[i]]
            i += 1
            while i < len(module_memory):
                if module_memory[i][0] >= memory_threshold:
                    bucket.append(module_memory[i])
                    i += 1
                else:
                    break
            new_module_memory += self.sort_by_name(bucket)
        # 优先级从低到高
        return new_module_memory

    
    def get_checkpoint_module(self, input_size, reduce_memory):
        module_memory = []
        max_activation = 0
        for name in self.levels_module["0"]:
            func = self.memory_predict_func[name]
            memory_size = func(input_size)
            max_activation = max(max_activation, memory_size)
            module_memory.append((memory_size, name))
        # module_memory.sort(reverse=True)
        module_memory = self.split_bucket(module_memory)

        module_set = set()
        # TODO: 需要判断最后一层是否使用checkpoint
        # reduce_memory += max_activation
        if reduce_memory <= 0:
            return module_set

        def get_fit_module(module_memory, target):
            for value in module_memory[::-1]:
                if value[0] >= target:
                    return value
            return module_memory[-1]

        curr_memory = 0
        candidate_modules = module_memory.copy()
        while curr_memory < reduce_memory and len(candidate_modules) > 0:
            entry = get_fit_module(candidate_modules, reduce_memory - curr_memory)
            curr_memory += entry[0]
            module_set.add(entry[1])
            candidate_modules.remove(entry)

        return module_set


class DoubleLevelGreedyStrategy(GreedyStrategy):
    # 只考虑 encoder 和 self-attention
    def get_checkpoint_module(self, input_size, reduce_memory):
        module_memory = []
        max_activation = 0
        for name in self.levels_module["0"]:
            func = self.memory_predict_func[name]
            memory_size = func(input_size)
            max_activation = max(max_activation, memory_size)
            module_memory.append((memory_size, name))
        # module_memory.sort(reverse=True)
        module_memory = self.split_bucket(module_memory)

        module_set = set()
        # TODO: 需要判断最后一层是否使用checkpoint
        # reduce_memory += max_activation
        if reduce_memory <= 0:
            return module_set

        def get_fit_module(module_memory, target):
            for value in module_memory[::-1]:
                if value[0] >= target:
                    return value
            return module_memory[-1]
        
        def use_sub_modules(reduce_memory, curr_memory, candidate_modules, entry, input_size):
            if entry == candidate_modules[-1] and entry[0] * 0.7 > reduce_memory - curr_memory:
                return sum([self.memory_predict_func[name + "-0-0"](input_size) for _, name in candidate_modules]) > reduce_memory - curr_memory
            return False

        curr_memory = 0
        candidate_modules = module_memory.copy()
        while curr_memory < reduce_memory and len(candidate_modules) > 0:
            entry = get_fit_module(candidate_modules, reduce_memory - curr_memory)
            if use_sub_modules(reduce_memory, curr_memory, candidate_modules, entry, input_size):
                # 如果需要减少的memory较少，则可以对self-attention使用checkpoint
                break
            else:
                curr_memory += entry[0]
                module_set.add(entry[1])
                candidate_modules.remove(entry)

        sub_candidate_modules = [(self.memory_predict_func[name + "-0-0"](input_size), name + "-0-0") for _, name in candidate_modules]
        while curr_memory < reduce_memory and len(sub_candidate_modules) > 0:
            entry = get_fit_module(sub_candidate_modules, reduce_memory - curr_memory)
            curr_memory += entry[0]
            module_set.add(entry[1])
            sub_candidate_modules.remove(entry)

        return module_set


class Manager:
    def __init__(self, warmup_iters=10):
        self.input_size = 0
        # module 映射
        self.modules = {}
        # 所有module name
        self.ordered_modules = []
        self.gc_layers = None

        self.iters = 0
        self.warmup_iters = warmup_iters
        
        self.max_levels = 0
        self.under_checkpoint = False

        # 存储 profile 相关信息，key = input_size, value = {"module": DataStorage}
        # 相同 input size 则叠加到一起
        self.data = {}
        self.checkpoint_count = 0

        self.checkpoint_module = set()
        self.non_checkpoint = set()
        self.checkpoint_history = {}

        self.memory_predict_func = {}
        self.strategy = None
        self.max_memory = 70 * (1024 ** 3)
        self.cached_strategy = {}
        self.static_strategy = False

        # 整个 model 的显存消耗
        self.model_memory_predict = None
        self.model_memory_data = {}
        self.checkpoint_reduce_memory = 0
        
        # 最大最小输入
        self.max_input = 142
        self.min_input = 32
        
        self.collector_overhead = []
        self.estimator_overhead = []
    
    def set_max_memory_GB(self, memory_threshold):
        self.max_memory = memory_threshold * (1024 ** 3)

    def register_module(self, name, module):
        """ 注册 module """
        self.modules[name] = module
        self.ordered_modules.append(name)
        self.max_levels = max(self.max_levels, name.count('-'))
    
    def set_non_checkpoint(self, name):
        """ 将 module 设为不可 checkpoint """
        self.non_checkpoint.add(name)
    
    def debug(self, local_checkpoint_module):
        print("="*20)
        print(f"memory budget: {self.max_memory / (1024 ** 2):.2f} MB")
        print(f"current memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"model memory: {self.model_memory_predict(self.input_size) / (1024 ** 2):.2f} MB")
        print(f"checkpoint modules")
        for module in local_checkpoint_module:
            print(f"\t{self.modules[module].__class__.__name__ + module}: {self.memory_predict_func[module](self.input_size) / (1024 ** 2):.2f} MB")
        print("="*20, flush=True)
    
    def round_input(self, input_size):
        interval = (self.max_input - self.min_input) // 50
        interval = max(1, interval)
        interval = 100
        # round_input_size = self.max_input - math.floor((self.max_input - input_size) / interval) * interval
        round_input_size = math.ceil(input_size / interval) * interval
        return round_input_size
    
    def register_gc_layers(self, gc_layers):
        gc_id = []
        for layer in gc_layers:
            for key, module in self.modules.items():
                if module == layer:
                    gc_id.append(key)
        self.gc_layers = gc_id

    def schedule_checkpoint(self):
        """ 计算需要进行checkpoint的module """
        local_checkpoint_module = set()

        if self.is_warmup():
            """ warmup 阶段的checkpoint module不变, 所以只需要在第一个iter设置即可 """
            if self.iters == 1:
                if self.gc_layers:
                    local_checkpoint_module = self.gc_layers
                else:
                    for key in self.modules:
                        if key.count('-') == self.max_levels // 2 and key.count('-') != 0 and key not in self.non_checkpoint:
                            local_checkpoint_module.add(key)
            else:
                local_checkpoint_module = self.checkpoint_module
        else:
            """ 正常训练阶段，从 cached_strategy 获得曾经的策略；如果没有，则计算策略 """
            round_input_size = self.round_input(self.input_size)
            if self.static_strategy:
                round_input_size = self.max_input
            if round_input_size in self.cached_strategy.keys():
                local_checkpoint_module = self.cached_strategy[round_input_size]
            else:
                available_memory = self.max_memory - torch.cuda.memory_allocated()
                reduce_memory = self.model_memory_predict(round_input_size) - available_memory
                if reduce_memory > 0:
                    t1 = time.time()
                    local_checkpoint_module = self.strategy.get_checkpoint_module(round_input_size, reduce_memory)
                    t2 = time.time()
                    self.estimator_overhead.append(t2 - t1)
                    # self.debug(local_checkpoint_module)
                    # torch.cuda.empty_cache()
                    torch.cuda.memory.reset_peak_memory_stats()
                self.cached_strategy[round_input_size] = local_checkpoint_module
        return local_checkpoint_module
    
    def init_strategy(self):
        self.strategy = GreedyStrategy(self.memory_predict_func, self.non_checkpoint, self.gc_layers)
        # self.strategy = DoubleLevelGreedyStrategy(self.memory_predict_func)
    
    def before_model_forward(self):
        self.iters += 1
        self.checkpoint_reduce_memory = -torch.cuda.memory_allocated()

        if self.warmup_finish():
            self.fit_memory_consume()
            self.init_strategy()

        self.checkpoint_count = 0
        self.checkpoint_module = self.schedule_checkpoint()
        if self.input_size not in self.data.keys():
            self.data[self.input_size] = {}

        # if self.is_warmup():
        #     torch.cuda.empty_cache()
        #     torch.cuda.memory.reset_peak_memory_stats()
        
        pass

    def collect_model_memory(self, memory):
        if self.input_size not in self.model_memory_data:
            self.model_memory_data[self.input_size] = DataStorage(time_use=0, mem_allocated=memory)
        else:
            self.model_memory_data[self.input_size].add(DataStorage(time_use=0, mem_allocated=memory))

    def after_forward(self):
        pass

    def after_update(self, memory_track=True):
        # 估计整个模型的 activation
        if self.is_warmup() and self.iters > 1:
            self.collect_model_memory(torch.cuda.max_memory_allocated() + self.checkpoint_reduce_memory)
            torch.cuda.empty_cache()
            torch.cuda.memory.reset_peak_memory_stats()
        # elif not self.is_warmup() and memory_track:
        #     torch.cuda.empty_cache()
        #     torch.cuda.memory.reset_peak_memory_stats()

    def set_input_size(self, input_size):
        self.input_size = input_size
        # import pdb; pdb.set_trace()
        self.before_model_forward()
    
    def is_warmup(self):
        return self.iters <= self.warmup_iters
    
    def warmup_finish(self):
        return self.iters == self.warmup_iters + 1

    def need_checkpoint(self, name):
        return name in self.checkpoint_module

    def prev_use_checkpoint(self):
        self.under_checkpoint = True
        self.checkpoint_count += 1
    
    def post_use_checkpoint(self):
        self.under_checkpoint = False
    
    def add_data(self, name, data):
        if name in self.data[self.input_size].keys():
            self.data[self.input_size][name].add(data)
        else:
            self.data[self.input_size][name] = data
    
    def get_data(self):
        """ Debug """
        ret = ""
        for key in self.ordered_modules:
            if key in self.data[self.input_size].keys():
                ret += f"{key}, {self.modules[key].__class__.__name__}: {str(self.data[self.input_size][key])}\n"
        return ret
    
    def print_all_data(self):
        for input_size, mem_map in self.data.items():
            print(f"input_size: {input_size}")
            for name, value in mem_map.items():
                print(f"{name}, {self.modules[name].__class__.__name__}: {value.get_memory()}")

    def serialize_data(self):
        output = {}
        for input_size, mem_map in self.data.items():
            tmp = {}
            for name, value in mem_map.items():
                tmp[name] = value.serialize()
            output[int(input_size)] = tmp
        return output
    
    def print_data(self):
        print(self.get_data())

    def fit_memory_consume(self):
        name2data = {} # {"module" : {"x": [], "y": []}}
        # collect data
        for input_size, data_map in self.data.items():
            for name, data_storage in data_map.items():
                if name not in name2data.keys():
                    name2data[name] = {"input_size": [], "memory": []}
                name2data[name]["input_size"] += [input_size] * len(data_storage.mem_allocated)
                name2data[name]["memory"] += data_storage.mem_allocated
        # fit
        for name, value in name2data.items():
            self.memory_predict_func[name] = PolyPrediction(value["input_size"], value["memory"])
            if not self.memory_predict_func[name].check_fit(self.memory_predict_func[name].poly_func, value["input_size"], value["memory"]):
                print(f"{name} dose not fit ploy")
        
        # do this for whole model
        input_size_list = []
        memory_list = []
        for input_size, data_storage in self.model_memory_data.items():
            input_size_list += [input_size] * len(data_storage.mem_allocated)
            memory_list += data_storage.mem_allocated
        self.model_memory_predict = PolyPrediction(input_size_list, memory_list)
        print("fit memory consume")
    
    def get_checkpoint_module(self):
        module_pair = []
        for key in self.checkpoint_module:
            module_pair.append((key, self.modules[key].__class__.__name__))
        return module_pair
