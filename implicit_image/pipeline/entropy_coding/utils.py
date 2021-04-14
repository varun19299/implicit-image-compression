import torch
import numpy as np
from typing import Union


def sparsity(tensor: Union[np.ndarray, torch.Tensor]):
    if isinstance(tensor, torch.Tensor):
        return (tensor == 0).sum() / tensor.numel()
    elif isinstance(tensor, np.ndarray):
        return (tensor == 0).sum() / tensor.size
