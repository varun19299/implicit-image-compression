from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from sparselearning.core import Masking
from sparselearning.funcs.decay import registry as decay_registry


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


def eval_epoch(
    eval_loader: DataLoader, model: Module, grid, img, **kwargs
) -> Tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        y_pred_full = torch.zeros_like(img)

        for h_batch, w_batch in eval_loader:
            x_test = grid[
                h_batch,
                w_batch,
            ]
            y_pred = model(x_test)
            y_pred_full[
                h_batch,
                w_batch,
            ] = y_pred

        test_loss = F.mse_loss(y_pred_full, img)

        test_PSNR = 10 * torch.log10(1 / test_loss)

    return y_pred_full, test_loss.item(), test_PSNR.item()


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device(device_str)
    else:
        return torch.device("cpu")


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
    if masking_cfg:
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
            input_size=(1, 2),
            density=masking_cfg.density,
            dense_gradients=masking_cfg.dense_gradients,
            sparse_init=masking_cfg.sparse_init,
            prune_mode=masking_cfg.prune_mode,
            growth_mode=masking_cfg.growth_mode,
            redistribution_mode=masking_cfg.redistribution_mode,
        )
        mask.add_module(model)

        return mask


def train_epoch(
    epoch: int,
    train_loader: DataLoader,
    model: Module,
    optim: Optimizer,
    lr_scheduler,
    grid,
    img,
    **kwargs
) -> float:
    # Unpack
    mask = kwargs.get("mask")
    pbar = kwargs.get("pbar")
    criterion = kwargs.get("criterion", F.mse_loss)

    iters = len(train_loader)

    # Single epoch
    for e, (h_slice, w_slice) in enumerate(train_loader):
        model.zero_grad()
        optim.zero_grad()

        x_train = grid[
            h_slice,
            w_slice,
        ]
        y_train = img[
            h_slice,
            w_slice,
        ]
        y_pred = model(x_train)

        # Any callable
        train_loss = criterion(
            y_pred,
            y_train,
        )
        train_loss.backward()

        stepper = mask if mask else optim
        stepper.step()
        lr_scheduler.step(epoch + e / iters)

        # Update pbar
        pbar.update(1)

    return train_loss.item()
