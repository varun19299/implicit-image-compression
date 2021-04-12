# Standard Libraries
import logging
import os

# Torch imports
import torch

# Other 3rd party imports
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from feathermap.feathernet import FeatherNet

# Modules
from models import registry as model_registry
from utils.catch_error import catch_error_decorator
from utils.data import load_img, get_grid
from utils.train_helper import (
    eval_epoch,
    get_device,
    setup_mask,
    train_epoch,
    get_optimizer_lr_scheduler,
)

from quant import context as quant_context
import encoding.zstandard


@catch_error_decorator
@hydra.main(config_name="config", config_path="conf")
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
        train_epoch(i, model, optim, grid, img, **train_kwargs)

        # Apply mask
        if mask and i <= cfg.masking.end_when:
            if i % cfg.masking.interval == 0:
                mask.update_connections()

        # Evaluate
        if (i + 1) % cfg.train.log_iters == 0:
            pred, test_loss, test_PSNR = eval_epoch(model, grid, img, **eval_kwargs)

            # pbar update
            msg = [
                f"Step: {i + 1}",
                f"loss: {test_loss:.4f}",
                f"PSNR: {test_PSNR:.3f}",
            ]

            if mask:
                msg += [f"Mask Prune Rate {mask.prune_rate:.5f}"]

            msg = " | ".join(msg)
            pbar.set_description(msg)
            logging.info(msg)

            # W&B logs
            if cfg.wandb.use:
                _log_dict = {
                    "loss": test_loss,
                    "PSNR": test_PSNR,
                    "image": [
                        wandb.Image(
                            pred.permute(2, 0, 1).detach(),
                            caption=cfg.img.name,
                        )
                    ],
                }
                if mask:
                    _log_dict.update(
                        {
                            "prune_rate": mask.prune_rate,
                            "density": mask.stats.total_density,
                        }
                    )
                wandb.log(_log_dict, step=i + 1)

    if cfg.quant:
        optim, lr_scheduler = get_optimizer_lr_scheduler(
            model, cfg.optim, quantize_mode=True
        )
        train_kwargs = {"lr_scheduler": lr_scheduler}

        # tqdm
        pbar = tqdm(total=cfg.quant.num_steps, dynamic_ncols=True)
        train_kwargs.update({"pbar": pbar, "mask": mask})  #

        with quant_context.Quantize(model, optim, cfg.quant) as q:
            for i in range(cfg.train.num_steps):
                train_epoch(i, model, optim, grid, img, **train_kwargs)

                # Evaluate
                if (i + 1) % cfg.train.log_iters == 0:
                    pred, compress_loss, compress_PSNR = eval_epoch(
                        model, grid, img, **eval_kwargs
                    )

                    # pbar update
                    msg = [
                        f"Compress step: {i + 1}",
                        f"Test loss: {compress_loss:.4f}",
                        f"Test PSNR: {compress_PSNR:.3f}",
                    ]
                    msg = " | ".join(msg)
                    pbar.set_description(msg)
                    logging.info(msg)

        quantized_model = q.convert()
        pred, compress_loss, compress_PSNR = eval_epoch(
            quantized_model, grid, img, **eval_kwargs
        )

        # W&B logs
        if cfg.wandb.use:
            _log_dict = {
                "compress_loss": compress_loss,
                "compress_PSNR": compress_PSNR,
                "compress_image": [
                    wandb.Image(
                        pred.permute(2, 0, 1).detach(),
                        caption=cfg.img.name,
                    )
                ],
            }
            wandb.log(_log_dict)

        # pbar update
        msg = [
            f"Eval Compress Step: {i + 1}",
            f"loss: {test_loss:.4f}",
            f"PSNR: {test_PSNR:.3f}",
        ]
        msg = " | ".join(msg)
        logging.info(msg)

    # Save weights
    if cfg.train.save_weights:
        state = {
            "state_dict": model.state_dict(),
        }
        torch.save(state, "model.pth")

        encoding.zstandard.compress_state_dict(quantized_model, "model.cpth", level=22)

    # Close wandb context
    if cfg.wandb.use:
        wandb.join()

    return test_PSNR


if __name__ == "__main__":
    main()
