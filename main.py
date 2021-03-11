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
from utils.data import get_dataloaders, load_img, get_grid
from utils.train_helper import eval_epoch, get_device, setup_mask, train_epoch


@catch_error_decorator
@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    device = get_device(cfg.device)

    # Image
    img = load_img(**cfg.img)

    # Grid (H x W x 2)
    grid = get_grid(cfg.img.height, cfg.img.width)

    # Slice loaders
    train_loader, eval_loader = get_dataloaders(
        img, cfg.train.batch_height, cfg.train.batch_width
    )

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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, eta_min=1e-5, T_max=cfg.train.num_steps, last_epoch=-1
    )

    # Setup mask if cfg.masking != {}
    mask = setup_mask(model, optim, cfg.get("masking"))

    # tqdm
    pbar = tqdm(total=cfg.train.num_steps, dynamic_ncols=True)

    for i in range(cfg.train.num_steps):
        train_loss = train_epoch(
            i,
            train_loader,
            model,
            optim,
            lr_scheduler,
            grid,
            img,
            dict(mask=mask, pbar=pbar),
        )

        # Apply mask
        if mask and i <= cfg.masking.end_when:
            if i % cfg.masking.interval == 0:
                mask.update_connections()

        # Evaluate
        if (i + 1) % cfg.train.log_iters == 0:
            y_pred_full, test_loss, test_PSNR = eval_epoch(
                eval_loader, model, grid, img
            )

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
                img_name = Path(cfg.img.path).name
                _log_dict = {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_PSNR": test_PSNR,
                    "image": [
                        wandb.Image(
                            y_pred_full.permute(2, 0, 1).detach(),
                            caption=img_name,
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

    return test_PSNR.item()


if __name__ == "__main__":
    main()
