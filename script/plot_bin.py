import re
from tkinter import N
import matplotlib.pyplot as plt

# f = open("../2022-06-26_21_50_08_swin_batchsize4_NoUseCheckpoint_mem.log", 'r')
f = open("test.log", 'r')

iter_f = iter(f)
data_bin = []

for line in iter_f:
  if "torch.Size" in line:
    num_list = re.findall(r"\d+",line)
    # print(num_list[2], num_list[3])
    num = int(num_list[2]) * int(num_list[3])
    data_bin.append(num)

# print(data_bin)
plt.hist(data_bin,bins=2)

plt.show()