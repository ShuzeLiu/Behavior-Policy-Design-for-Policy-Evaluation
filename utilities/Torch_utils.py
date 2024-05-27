from utilities.Config import *
import torch
import numpy as np
import os
import nvidia_smi




def tensor(x, target_type, device):
    if isinstance(x, torch.Tensor): 
        return x
    x = np.asarray(x, dtype=target_type)
    x = torch.from_numpy(x).to(device)
    return x

    
def tensor_to_np(x):
    return x.cpu().detach().numpy()

def set_n_thread(n):
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    torch.set_num_threads(n)


def best_gpu():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    list_free = []
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
        list_free.append(100*info.free/info.total)
    nvidia_smi.nvmlShutdown()
    return np.argmax(list_free)