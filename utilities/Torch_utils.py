from utilities.Config import *
import torch
import numpy as np




def tensor(x, target_type, device):
    if isinstance(x, torch.Tensor): 
        return x
    x = np.asarray(x, dtype=target_type)
    x = torch.from_numpy(x).to(device)
    return x

def tensor_to_np(x):
    return x.cpu().detach().numpy()
