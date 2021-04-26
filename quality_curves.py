import os, sys
os.environ['OPENCV_IO_ENABLE_JASPER']='True'
import cv2
import hydra
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import time
from ..data import load_img
#import os, sys
#sys.path.append(os.path.abspath('../implicit_image'))
from data import load_img
import json

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
    plt.tight_layout()
    plt.savefig(
        path / f"jpeg_{y_key}_vs_{x_key}.pdf",
        dpi=150,
    )
    plt.show()


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg: DictConfig):

    jpeg_plot(cfg, "flower_foveon.ppm")
    #jpeg_2000_plot(cfg,"flower_foveon.ppm")
    #webp_plot(cfg, "flower_foveon.ppm")


def jpeg_2000_plot(cfg, img_name):

    # Initialize directories
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
    path.mkdir(exist_ok=True, parents=True)
    dump_path = Path("/tmp/jpeg_dump")
    dump_path.mkdir(exist_ok=True, parents=True)

    single_cfg = cfg.copy()
    single_cfg.img.path = str(Path(single_cfg.img.path).parent / img_name)

    img = load_img(**single_cfg.img)  # H x W x 3
    img = (img * 255.0).numpy()[:, :, ::-1]
    
    #specifies the sizes the jpg2000 will compress to
    rates = np.linspace(100, 300, 20)

    img_dict = {}
    psnr_ll = []
    size_ll = []

    for rate in rates:
        print("rate: " + str(rate))
        print(type(img))
        image_pil = Image.fromarray(img,"RGB")

        image_pil.save("test2.jpg")

        image_pil.save('test_name.jp2', quality_mode = "rates", quality_layers = [rate])

        decimg = Image.open('test_name.jp2')

        r, g, b = decimg.split()
        decimg = Image.merge("RGB", (r, g, b))
        data = np.asarray(decimg)


        dump_file = dump_path / f"{img_name}_quality_{rate}.jp2"
        cv2.imwrite(str(dump_file), data)

        psnr = 10 * np.log10(255 ** 2 / ((data - img) ** 2).mean())
        psnr_ll.append(psnr)
        print(psnr)
        time.sleep(0.2)
        size = dump_file.stat().st_size
        size_ll.append(size // 1024)
        print(size)
    img_dict[img_name] = {"psnr": psnr_ll, "size": size_ll}

    # Convert to list
    for name in img_dict:
        for metric, metric_ll in img_dict[name].items():
            if isinstance(metric_ll, np.ndarray):
                img_dict[name][metric] = metric_ll.tolist()

    with open(path.parent / "csv" / "jpeg_dump.json", "w") as f:
        json.dump(img_dict, f, indent=4, sort_keys=True)


def webp_plot(cfg, img_name):

    # Initialize directories
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
    path.mkdir(exist_ok=True, parents=True)
    dump_path = Path("/tmp/jpeg_dump")
    dump_path.mkdir(exist_ok=True, parents=True)

    single_cfg = cfg.copy()
    single_cfg.img.path = str(Path(single_cfg.img.path).parent / img_name)

    img = load_img(**single_cfg.img)  # H x W x 3
    img = (img * 255.0).numpy()[:, :, ::-1]
    
    #specifies the sizes the jpg2000 will compress to
    rates = np.linspace(1, 100, 20)

    img_dict = {}
    psnr_ll = []
    size_ll = []

    for rate in rates:
        print("rate: " + str(rate))
        print(type(img))
        image_pil = Image.fromarray(img, "RGB")
 
        image_pil.save('test_name.webp', quality = rate)

        #image_pil.save("test.jpg")

        decimg = Image.open('test_name.webp')

        r, g, b = decimg.split()
        decimg = Image.merge("RGB", (r, g, b))
        data = np.asarray(decimg)


        dump_file = dump_path / f"{img_name}_quality_{rate}.webp"
        cv2.imwrite(str(dump_file), data)

        psnr = 10 * np.log10(255 ** 2 / ((data - img) ** 2).mean())
        psnr_ll.append(psnr)
        print(psnr)
        time.sleep(0.2)
        size = dump_file.stat().st_size
        size_ll.append(size // 1024)
        print(size)
    img_dict[img_name] = {"psnr": psnr_ll, "size": size_ll}

    # Convert to list
    for name in img_dict:
        for metric, metric_ll in img_dict[name].items():
            if isinstance(metric_ll, np.ndarray):
                img_dict[name][metric] = metric_ll.tolist()

    with open(path.parent / "csv" / "jpeg_dump.json", "w") as f:
        json.dump(img_dict, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
        




