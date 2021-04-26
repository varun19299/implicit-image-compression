# Standard Libraries
import logging
import os

# Torch imports
import torch
from typing import Dict

# Other 3rd party imports
import hydra
import wandb
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from implicit_image.pipeline.feathermap import FeatherNet
from implicit_image.pipeline.quant import context as quant_context
from implicit_image.pipeline import entropy_coding

# Modules
from implicit_image.models import registry as model_registry
from implicit_image.utils.catch_error import catch_error_decorator
from implicit_image.data import load_img, get_grid
from implicit_image.utils.train_helper import (
    eval_epoch,
    get_device,
    setup_mask,
    train_epoch,
    get_optimizer_lr_scheduler,
)


def file_and_wandb_logger(
    label: str, step: int, log_dict: Dict, pbar: tqdm = None, use_wandb: bool = False
):
    """
    Log to pbar, file and W&B
    """
    msg = [f"{k}: {v:.4f}" for k, v in log_dict.items() if isinstance(v, float)]
    msg = [label, f"Step: {step}"] + msg
    msg = " | ".join(msg)

    logging.info(msg)

    if pbar:
        pbar.set_description(msg)

    if use_wandb:
        wandb.log(log_dict, step=step)


@catch_error_decorator
@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    device = get_device(cfg.device)

    # Image (H x W x 3)
    img = load_img(**cfg.img)

    # Grid (H x W x 2)
    grid = get_grid(cfg.img.height, cfg.img.width)

    # Construct composed (Siren or MLP)
    _, _, c = grid.shape
    logging.info(f"Grid of shape {grid.shape}")

    # Small Dense: lower params with a "narrow" MLP
    _small_density = 1.0
    if cfg.get("masking") and cfg.masking.name == "Small_Dense":
        _small_density = cfg.masking.density
    model = model_registry[cfg.mlp.name](**cfg.mlp, small_dense_density=_small_density)

    # Feathermap: hashing based compression
    if cfg.get("masking") and cfg.masking.name == "Feathermap":
        model = FeatherNet(model, compress=cfg.masking.density)

    # Send to device
    model = model.to(device)
    grid = grid.to(device)
    img = img.to(device)

    # wandb
    if cfg.wandb.use:
        with open(cfg.wandb.api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read()

        wandb.init(
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            reinit=True,
            save_code=True,
        )
        wandb.watch(model)

    # Train
    model.train()
    optim, lr_scheduler = get_optimizer_lr_scheduler(model, cfg.optim)
    train_kwargs = {"lr_scheduler": lr_scheduler}
    eval_kwargs = {}

    # Training multiplier
    training_multiplier = cfg.train.multiplier
    cfg.train.num_steps *= training_multiplier

    if cfg.get("masking"):
        if cfg.masking.get("end_when"):
            cfg.masking.end_when *= training_multiplier
            cfg.masking.end_when = int(cfg.masking.end_when)

        if cfg.masking.get("interval"):
            cfg.masking.interval *= training_multiplier
            cfg.masking.interval = int(cfg.masking.interval)

    # tqdm
    pbar = tqdm(total=cfg.train.num_steps, dynamic_ncols=True)
    train_kwargs.update({"pbar": pbar})

    # Setup mask if cfg.masking != {}
    mask = setup_mask(model, optim, cfg.get("masking"))
    train_kwargs.update({"mask": mask})

    # AMP
    if cfg.train.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        context = torch.cuda.amp.autocast
        train_kwargs.update({"scaler": scaler, "context": context})
        eval_kwargs.update({"context": context})

    for i in range(cfg.train.num_steps):
        train_epoch(model, optim, grid, img, **train_kwargs)

        # Apply mask
        if mask and i <= cfg.masking.end_when:
            if i % cfg.masking.interval == 0:
                mask.update_connections()

        # Evaluate
        if (i + 1) % cfg.train.log_steps == 0:
            pred, test_loss, test_PSNR, test_PSNR_8bit = eval_epoch(
                model, grid, img, **eval_kwargs
            )

            log_dict = {
                "loss": test_loss,
                "PSNR": test_PSNR,
                "PSNR_8bit": test_PSNR_8bit,
                "image": [
                    wandb.Image(
                        pred.permute(2, 0, 1).detach(),
                        caption=cfg.img.name,
                    )
                ],
            }
            if mask:
                log_dict.update(
                    {"Prune Rate": mask.prune_rate, "Density": mask.stats.total_density}
                )

            # Pbar and file logging
            file_and_wandb_logger(
                "Train", i + 1, log_dict, pbar, use_wandb=cfg.wandb.use
            )

    if cfg.quant:
        # Model
        quantized_model = deepcopy(model)

        optim, lr_scheduler = get_optimizer_lr_scheduler(
            quantized_model, cfg.optim, quantize_mode=True
        )
        eval_kwargs = {}
        train_kwargs = {"lr_scheduler": lr_scheduler}

        # tqdm
        pbar = tqdm(total=cfg.quant.num_steps, dynamic_ncols=True)
        train_kwargs.update({"pbar": pbar, "mask": mask})

        with quant_context.Quantize(quantized_model, optim, cfg.quant) as q:
            for i in range(cfg.quant.num_steps):
                train_epoch(quantized_model, optim, grid, img, **train_kwargs)

                # Evaluate
                if (i + 1) % cfg.quant.log_steps == 0:
                    pred, compress_loss, compress_PSNR, compress_PSNR_8bit = eval_epoch(
                        quantized_model, grid, img, **eval_kwargs
                    )

                    # Pbar and file logging
                    log_dict = {
                        "loss": compress_loss,
                        "PSNR": compress_PSNR,
                        "PSNR_8bit": compress_PSNR_8bit,
                    }

                    # Pbar and file logging
                    file_and_wandb_logger("Quant", i + 1, log_dict, pbar)

        # Evaluate final model
        quantized_model = q.convert()
        pred, compress_loss, compress_PSNR, compress_PSNR_8bit = eval_epoch(
            quantized_model, grid, img, **eval_kwargs
        )

        log_dict = {
            "Quant loss": compress_loss,
            "Quant PSNR": compress_PSNR,
            "Quant PSNR 8bit": compress_PSNR_8bit,
        }

        # Pbar and file logging
        msg = [
            f"Post Quant",
            f"Train step: {cfg.train.num_steps}",
            f"Quant step: {cfg.quant.num_steps}",
        ]
        msg += [f"{k}:{v:.4f}" for k, v in log_dict.items()]
        msg = " | ".join(msg)
        logging.info(msg)

        # W&B logs
        if cfg.wandb.use:
            log_dict.update(
                {
                    "Quant image": [
                        wandb.Image(
                            pred.permute(2, 0, 1).detach(),
                            caption=cfg.img.name,
                        )
                    ],
                }
            )
            wandb.log(log_dict, step=cfg.train.num_steps)

    # Save weights
    if cfg.train.save_weights:
        torch.save({"state_dict": model.state_dict()}, "model.pth")

        if cfg.train.mixed_precision:
            torch.save({"state_dict": model.half().state_dict()}, "model_half.pth")

        if cfg.quant:
            quantized_model = quantized_model.half()

            if cfg.entropy_coding:
                compressed_bytes = entropy_coding.compress_state_dict(
                    quantized_model, "model_quantized", **cfg.entropy_coding
                )
                msg = f"Compressed bytes {compressed_bytes}"
                print(msg)
                logging.info(msg)

                if cfg.wandb.use:
                    wandb.log(
                        {"Compressed Bytes": compressed_bytes}, step=cfg.train.num_steps
                    )

    # Close wandb context
    if cfg.wandb.use:
        wandb.join()

    return test_PSNR, compressed_bytes


if __name__ == "__main__":
    main()
