# Libraries
import hydra
import logging
from models.siren import Siren
import os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Dynamic Sparsity
from sparselearning.core import Masking
from sparselearning.funcs.decay import registry as decay_registry

from feathermap.feathernet import FeatherNet

# Torch imports
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import registry as model_registry
from utils.catch_error import catch_error_decorator
from utils.train_utils import (
    load_img,
    compress_indices,
    get_dataloaders,
    get_device,
    get_grid,
)
import wandb


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
    train_loader, test_loader = get_dataloaders(
        img, cfg.train.batch_height, cfg.train.batch_width
    )

    # Construct composed (Siren or MLP)
    _, _, c = grid.shape
    logging.info(f"Encoded grid of shape {grid.shape}")

    model = model_registry[cfg.mlp.name](**cfg.mlp)
    # model = FeatherNet(model, compress=0.5)

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

    # Setup mask
    mask = None
    if cfg.get("masking"):
        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "T_max": cfg.masking.end_when,
                "T_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }
        else:
            kwargs = {"prune_rate": cfg.masking.prune_rate, "T_max": cfg.masking.end_when}

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        mask = Masking(
            optim,
            decay,
            input_size=(1,2),
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )
        mask.add_module(model)

    # tqdm
    pbar = tqdm(total=cfg.train.num_steps, dynamic_ncols=True)
    iters = len(train_loader)

    for i in range(cfg.train.num_steps):
        for e, (h_batch, w_batch) in enumerate(train_loader):
            model.zero_grad()
            optim.zero_grad()

            x_train = grid[
                h_batch,
                w_batch,
            ]
            y_train = img[
                h_batch,
                w_batch,
            ]
            y_pred = model(x_train)

            train_loss = F.mse_loss(
                y_pred,
                y_train,
            )
            train_loss.backward()

            stepper = mask if mask else optim
            stepper.step()
            lr_scheduler.step(i + e / iters)

            # Update pbar
            pbar.update(1)

        # Apply mask
        if mask and i <= cfg.masking.end_when:
            if i % cfg.masking.interval == 0:
                mask.update_connections()

        if (i + 1) % cfg.train.log_iters == 0:
            with torch.no_grad():
                y_pred_full = torch.zeros_like(img)

                for h_batch, w_batch in test_loader:
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

                # pbar update
                msg = f"Step: {i + 1} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test PSNR: {test_PSNR:.3f}"

                if mask:
                    msg += f" | Mask Prune Rate {mask.prune_rate:.5f}"
                pbar.set_description(msg)
                logging.info(msg)

                if cfg.wandb.use:
                    # W&B logs
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
                        _log_dict = {
                            **_log_dict,
                            "prune_rate": mask.prune_rate,
                            "density": mask.stats.total_density,
                        }
                    wandb.log(_log_dict, step=i + 1)

    # Save weights
    if cfg.train.save_weights:
        state = {
            "state_dict": compress_indices(model.state_dict()),
        }
        torch.save(state, "model.pth")

    # Close wandb context
    if cfg.wandb.use:
        wandb.join()

    return test_PSNR.item()


if __name__ == "__main__":
    main()