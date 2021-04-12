from contextlib import contextmanager
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from torch.nn import Module, functional as F
from torch.optim import Optimizer

from sparselearning.core import Masking
from sparselearning.funcs.decay import registry as decay_registry


@contextmanager
def _blank_context():
    yield


def compress_indices(state_dict: Dict) -> Dict:
    """
    Compresses index tensors of sparse matrices (using int8/16/32).
    :param state_dict: MLPs state dict
    :return: a compressed version
    """

    def highest_precision(tensor: torch.Tensor):
        if tensor.max() < 2 ** 8:
            return torch.int8
        elif tensor.max() < 2 ** 16:
            return torch.int16
        elif tensor.max() < 2 ** 32:
            return torch.int32
        else:
            return torch.int64

    for key in state_dict.keys():
        if "indices" in key:
            state_dict[key] = state_dict[key].to(highest_precision(state_dict[key]))

    return state_dict


@torch.no_grad()
def eval_epoch(model: Module, grid, img, **kwargs) -> Tuple[torch.Tensor, float, float]:
    # Automatic mixed precision
    context = kwargs.get("criterion", _blank_context)

    model.eval()

    with context():
        pred = model(grid)

    test_loss = F.mse_loss(pred, img)
    test_PSNR = 10 * torch.log10(1 / test_loss)

    return pred, test_loss.item(), test_PSNR.item()


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device(device_str)
    else:
        return torch.device("cpu")


def get_optimizer_lr_scheduler(
    model: Module, optim_cfg: Dict, quantize_mode: bool = False
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optim_dict = {
        "adam": torch.optim.Adam,
    }

    optim = optim_dict[optim_cfg.name](model.parameters(), lr=optim_cfg.learning_rate)

    if quantize_mode:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1000, gamma=0.5)
    else:
        # empirically, step lr (cut by 5 after 1k steps) should be better
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 2000, gamma=0.5)

    return optim, lr_scheduler


def setup_mask(
    model: Module, optim: Optimizer, masking_cfg: DictConfig = None
) -> Masking:
    """
    Get masking instance that wraps model
    :param model: pytorch Model
    :param optim: optimizer
    :param masking_cfg: cfg.masking (if present)
    :return: Masking instance
    """
    if masking_cfg and not masking_cfg.dense:
        if masking_cfg.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - masking_cfg.final_density,
                "T_max": masking_cfg.end_when,
                "T_start": masking_cfg.start_when,
                "interval": masking_cfg.interval,
            }
        else:
            kwargs = {
                "prune_rate": masking_cfg.prune_rate,
                "T_max": masking_cfg.end_when,
            }

        decay = decay_registry[masking_cfg.decay_schedule](**kwargs)

        mask = Masking(
            optim,
            decay,
            input_size=(1, 1, 2),
            density=masking_cfg.density,
            dense_gradients=masking_cfg.dense_gradients,
            sparse_init=masking_cfg.sparse_init,
            prune_mode=masking_cfg.prune_mode,
            growth_mode=masking_cfg.growth_mode,
            redistribution_mode=masking_cfg.redistribution_mode,
        )
        model.train()
        mask.add_module(model)

        return mask


def train_epoch(
    epoch: int, model: Module, optim: Optimizer, grid, img, **kwargs
) -> float:
    # Unpack
    mask: Masking = kwargs.get("mask")
    pbar = kwargs.get("pbar")
    lr_scheduler = kwargs.get("lr_scheduler")
    criterion = kwargs.get("criterion", F.mse_loss)

    # Automatic mixed precision
    context = kwargs.get("criterion", _blank_context)
    scaler = kwargs.get("scaler")

    model.train()
    optim.zero_grad()

    # Forward pass
    with context():
        pred = model(grid)
        # Any callable
        train_loss = criterion(
            pred,
            img,
        )

    if scaler:
        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(train_loss).backward()
    else:
        train_loss.backward()

    if mask:
        # If mask, pass the scalar to it
        mask.step(scaler)
    else:
        if scaler:
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optim)
            # Updates the scale for next iteration
            scaler.update()
        else:
            optim.step()

    # Update pbar
    pbar.update(1)

    if lr_scheduler:
        lr_scheduler.step()
    return train_loss.item()
