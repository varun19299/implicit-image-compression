"""
Plot PSNR, Storage as a function of JPEG quality
"""
import cv2
import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import time
from utils.data import load_img
from hurry.filesize import size as hurry_size


def _plot_image_dict(img_dict, x_key: str, y_key: str, path: Path):
    _label_dict = {"psnr": "PSNR (in dB)", "quality": "Quality", "size": "Size (in KB)"}
    # plt.figure(figsize=(8, 8))
    for img_name in img_dict:
        plt.plot(
            img_dict[img_name][x_key],
            img_dict[img_name][y_key],
            linestyle="--",
            marker="o",
        )

    plt.xlabel(_label_dict[x_key])
    plt.ylabel(_label_dict[y_key])
    # plt.title(f"JPEG {y_key.capitalize()} vs {x_key.capitalize()}")
    plt.legend(img_dict.keys())
    plt.grid()
    plt.savefig(
        path / f"jpeg_{y_key}_vs_{x_key}.pdf",
        dpi=150,
    )
    plt.show()


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    img_names = ["flower_foveon.ppm", "big_building.ppm", "bridge.ppm"]
    img_dict = {}
    quality_ll = np.linspace(1, 100, 20)

    # Initialize directories
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
    path.mkdir(exist_ok=True, parents=True)

    for img_name in img_names:
        single_cfg = cfg.copy()
        single_cfg.img.path = str(Path(single_cfg.img.path).parent / img_name)

        # Load normalised image
        img = load_img(**single_cfg.img)  # H x W x 3
        img = (img * 255.0).numpy()[:, :, ::-1]

        img_name = Path(single_cfg.img.path).stem

        dump_path = Path("/tmp/jpeg_dump")
        dump_path.mkdir(exist_ok=True, parents=True)

        psnr_ll = []
        size_ll = []

        for quality in quality_ll:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode(".jpg", img, encode_param)
            assert result, f"Could not encode image at quality {quality}"

            # decode from jpeg format
            decimg = cv2.imdecode(encimg, 1)

            dump_file = dump_path / f"{img_name}_quality_{quality}.jpg"

            cv2.imwrite(str(dump_file), decimg)

            psnr = 10 * np.log10(255 ** 2 / ((decimg - img) ** 2).mean())
            psnr_ll.append(psnr)

            time.sleep(0.2)
            size = dump_file.stat().st_size
            size_ll.append(size // 1024)

        img_dict[img_name] = {"psnr": psnr_ll, "size": size_ll, "quality": quality_ll}

    _plot_image_dict(img_dict, "quality", "psnr", path)
    _plot_image_dict(img_dict, "quality", "size", path)
    _plot_image_dict(img_dict, "size", "psnr", path)


if __name__ == "__main__":
    main()
