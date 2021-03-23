# Standard Libraries
import logging
import os
from pathlib import Path

# Other 3rd party imports
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Torch imports
import torch
import torch.nn.functional as F

# Modules
from models import registry as model_registry
from utils.catch_error import catch_error_decorator
from utils.data import load_img, get_grid
from utils.train_helper import eval_epoch, get_device, setup_mask, train_epoch


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

    model = model_registry[cfg.mlp.name](**cfg.mlp)

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

    # Training multiplier
    training_multiplier = cfg.train.multiplier
    cfg.train.num_steps *= training_multiplier

    if cfg.get("masking"):
        cfg.masking.end_when *= training_multiplier
        cfg.masking.end_when = int(cfg.masking.end_when)

        cfg.masking.interval *= training_multiplier
        cfg.masking.interval = int(cfg.masking.interval)

    # Train
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # empirically, step lr (cut by 5 after 1k steps) should be better
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 2000, gamma=0.5)
    train_kwargs = {"lr_scheduler": lr_scheduler}
    eval_kwargs = {}

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
        train_loss = train_epoch(
            i,
            model,
            optim,
            grid,
            img,
            **train_kwargs,
        )

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
                f"Train loss: {train_loss:.4f}",
                f"Test loss: {test_loss:.4f}",
                f"Test PSNR: {test_PSNR:.3f}",
            ]

            if mask:
                msg += [f"Mask Prune Rate {mask.prune_rate:.5f}"]

            msg = " | ".join(msg)
            pbar.set_description(msg)
            logging.info(msg)

            # W&B logs
            if cfg.wandb.use:
                _log_dict = {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_PSNR": test_PSNR,
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

    # Save weights
    if cfg.train.save_weights:
        state = {
            "state_dict": model.state_dict(),
        }
        torch.save(state, "model.pth")

    # Close wandb context
    if cfg.wandb.use:
        wandb.join()

    return test_PSNR


if __name__ == "__main__":
    main()
