from typing import DefaultDict
import torch
from mmcv.runner import OptimizerHook, HOOKS
try:
    import apex
except:
    print('apex is not installed')

import torch
import json
import time
from .manager import Manager, cast_forward, recover_forward



@HOOKS.register_module()
class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training."""

    def __init__(self, update_interval=1, grad_clip=None, coalesce=True, bucket_size_mb=-1, use_fp16=False, **kwargs):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16

        memory_threshold = kwargs.get("memory_threshold", 16)
        torch.cuda.set_per_process_memory_fraction(memory_threshold * (1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)
        print(f'Memory threshold: {memory_threshold} GiB')
        dc_cfg = kwargs.get("dc", {"enable": False})
        if dc_cfg["enable"]:
            self.dc_manager = Manager(dc_cfg.get("warmup_iters", 30))
            self.dc_manager.set_max_memory_GB(memory_threshold=memory_threshold - kwargs.get("memory_buffer", 0.5))
            print(f'dc manager max memory GB: {memory_threshold - kwargs.get("memory_buffer", 0.5)}')
            if dc_cfg.get("static", False):
                print('Static strategy DC (sublinear)')
                self.dc_manager.static_strategy = True
                self.dc_manager.max_input = dc_cfg["max_input"]
        else:
            self.dc_manager = None

        self.shape_memory = []
        self.start_training_time = 0

        self.input_shape_list = []
        self.mem_list = []
        self.time_list = [0.0]

        self.input_shape = DefaultDict(int)
        self.memory_collect = {}
        self.shape_order = []

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def before_train_epoch(self, runner):
        self.start_training_time = time.time()
        if self.dc_manager:
            cast_forward(runner.model.module.backbone, "0", self.dc_manager)
            runner.model.module.dc_manager = self.dc_manager
            if hasattr(runner.model.module.backbone, "gc_layers"):
                self.dc_manager.register_gc_layers(runner.model.module.backbone.gc_layers)

    def after_train_epoch(self, runner):
        print("shape_count=" + json.dumps(self.input_shape))
        print("memory_count=" + json.dumps(self.memory_collect))
        print("shape_order=" + json.dumps(self.shape_order))

        if self.dc_manager:
            print("collector_overhead=" + json.dumps(self.dc_manager.collector_overhead))
            print(f"collector_overhead: sum={sum(self.dc_manager.collector_overhead)}, cnt={len(self.dc_manager.collector_overhead)}")
            print("estimator_overhead=" + json.dumps(self.dc_manager.estimator_overhead))
            print(f"estimator_overhead: sum={sum(self.dc_manager.estimator_overhead)}, cnt={len(self.dc_manager.estimator_overhead)}")


        print(f"training time: {(time.time() - self.start_training_time) / 60:.2f} min", flush=True)
        print("shape_memory=" + json.dumps(self.shape_memory), flush=True)
        if self.dc_manager:
            recover_forward(runner.model.module.backbone)

    def before_train_iter(self, runner):
        torch.cuda.memory.reset_peak_memory_stats()

    def after_train_iter(self, runner):
        seq_length = runner.model.module.input_shape[-1] * runner.model.module.input_shape[-2]
        self.input_shape[seq_length] += 1
        self.shape_order.append(seq_length)
        if seq_length not in self.memory_collect:
            self.memory_collect[seq_length] = []
        self.memory_collect[seq_length].append(torch.cuda.max_memory_allocated())

        print("##################################### input shape #####################################")
        self.input_shape_list.append(runner.model.module.input_shape)
        print(self.input_shape_list[-1])

        print("##################################### peak mem #####################################")
        self.mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        print("{}MB".format(self.mem_list[-1]))

        print("##################################### batch time #####################################")
        self.time_list.append(time.time())
        print("{}ms".format((self.time_list[-1] - self.time_list[-2]) * 1000))

        runner.outputs['loss'] /= self.update_interval
        if self.use_fp16:
            with apex.amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # TODO: backward
            runner.outputs['loss'].backward()
            # runner.outputs['loss'].decheckpoint()
            # torch.clear_checkpointpool()
        if self.every_n_iters(runner, self.update_interval):
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()
            runner.optimizer.zero_grad()
        # TODO: 显存峰值
        self.shape_memory.append((list(runner.model.module.input_shape), torch.cuda.max_memory_allocated()))
        if self.dc_manager:
            self.dc_manager.after_update()

        if (len(self.shape_memory) == 500):
            self.after_train_epoch(runner)
            exit()
        # torch.cuda.reset_peak_memory_stats()
        # print(f"mem_data: {mem_data}".format(mem_data))
        # f = open('/home/hjw/program/Swin-Transformer-Object-Detection/dateset/exp/test', 'a+')
        # f.write(str(mem_data))
        # f.write(", \n")
