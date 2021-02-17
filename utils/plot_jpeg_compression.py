"""
Plot JPEG PSNR, SSIM vs quality
"""

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    img_path = Path(cfg.img.path)
    if img_path.is_dir():
        logging.info(f"Running on images from directory {img_path}")
        single_img_cfg = cfg.copy()

        # Metrics
        test_PSNR_ll = []

        file_ll = img_path.glob("*")
        for file in file_ll:
            logging.info(f"File {file}")

            # Single image config
            single_img_cfg.img.path = str(file)
            single_img_cfg.use_wandb = False

            test_PSNR_ll.append(run_single_image(single_img_cfg))

        test_PSNR_ll = torch.tensor(test_PSNR_ll)
        test_PSNR = test_PSNR_ll.mean()
    else:
        test_PSNR = run_single_image(cfg)

    logging.info(f"Overall PSNR on {img_path.parent/ img_path.name}:{test_PSNR:.3f}")
    return test_PSNR.item()


if __name__ == "__main__":
    main()
