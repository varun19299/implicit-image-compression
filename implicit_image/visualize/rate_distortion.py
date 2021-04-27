"""
Run as:

 python implicit_image/visualize/rate_distortion.py wandb.project=finals_simple img=flower

or:

 make vis.rate_distortion.flower etc.

See conf/img for names of img YAML files.
"""
import itertools
import logging
import os
from collections import namedtuple
from pathlib import Path
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from implicit_image.data import load_img

# Matplotlib font sizes
TINY_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def ours_rate_distortion(
    runs,
    image: str,
    density_ll: List[float] = [0.1],
    suffix_ll: List[str] = [""],
    reorder: bool = True,
) -> pd.DataFrame:
    """
    Get stats saved on W&B.

    List all possible choices for (image, masking, density, suffix).

    We'll try matching the exhaustive caretesian product of
    (masking_ll x init_ll x suffix_ll x density_ll etc).

    :param runs: Experiment run
    :param image: name
    :param density_ll: List of density values (1 - sparsity)
    :param suffix_ll: List of suffixes
    :param reorder: sort methods alphabetically

    :return: Dataframe containing test accuracies of methods
    :rtype: pd.DataFrame
    """
    columns = ["Density", "Bytes", "PSNR"]
    df = pd.DataFrame(columns=columns)

    # Pre-process
    logging.info("Grouping runs by name")
    run_dict = {}
    for run in runs:
        run_dict[run.name] = run

    for e, (density, suffix) in enumerate(itertools.product(density_ll, suffix_ll)):
        tags = [image, suffix, f"density_{round(density,2)}"]

        # Remove tags that are None
        name = "_".join([str(tag) for tag in tags if tag])
        logging.debug(name)

        run = run_dict.get(name, None)
        if not run:
            logging.debug(f"Run {name} not present")
            continue

        compressed_psnr = run.summary.get("Quant PSNR")
        compressed_bytes = run.summary.get("Compressed Bytes")
        df.loc[e] = [density, compressed_bytes, compressed_psnr]

    df = df.dropna()
    df["Bytes"] = df["Bytes"].astype(int)
    df = df.sort_values(by=["Density", "Bytes"])

    if reorder:
        df = df.reset_index(drop=True)

    return df


def traditional_rate_distortion(img, img_name: str, extension="jpg"):
    # Initialize directories
    dump_path = Path("/tmp/traditional")
    dump_path.mkdir(exist_ok=True, parents=True)

    # specifies the sizes the jpg2000 will compress to
    quality_ll = np.linspace(0, 100, 20).astype(int).tolist()

    psnr_ll = []
    size_ll = []

    extension_flags = {
        "jpg": cv2.IMWRITE_JPEG_QUALITY,
        "webp": cv2.IMWRITE_WEBP_QUALITY,
        "jp2": cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,
    }

    img = (img * 255)[:, :, ::-1].astype(int)

    for quality in quality_ll:
        dump_file = dump_path / f"{img_name}_{quality}.{extension}"

        if extension == "webp":
            img_pil = Image.fromarray(img[:, :, ::-1].astype(np.uint8))
            img_pil.save(dump_file, quality=quality)
            decimg = np.asarray(Image.open(dump_file))[:, :, ::-1]
        else:
            encode_param = [int(extension_flags[extension]), quality]
            result, encimg = cv2.imencode(f".{extension}", img, encode_param)
            assert result, f"Could not encode image at quality {quality}"

            # decode
            decimg = cv2.imdecode(encimg, -1)
            cv2.imwrite(str(dump_file), decimg)

        psnr = 10 * np.log10(255 ** 2 / ((decimg.astype(int) - img) ** 2).mean())
        psnr_ll.append(psnr)

        size = dump_file.stat().st_size
        size_ll.append(size)

    df = pd.DataFrame({"PSNR": psnr_ll, "Bytes": size_ll})
    df = df.sort_values(by=["Bytes"])
    df = df.reset_index(drop=True)

    return df


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    # Get runs
    api = wandb.Api()
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")

    density_ll = [0.01, 0.02] + np.arange(0.05, 0.95, step=0.05).tolist()
    ours_df = ours_rate_distortion(runs, cfg.img.name, density_ll=density_ll)

    # Open image
    # H x W x 3
    # RGB between 0...255
    img = load_img(**cfg.img).numpy()

    PlotProperty = namedtuple("PlotProperty", ["name", "color", "linewidth", "alpha"])

    linewidth = 2
    alpha = 0.8
    plot_dict = {
        "jpg": PlotProperty("JPEG", "orange", linewidth, alpha),
        "webp": PlotProperty("WebP", "blue", linewidth, alpha),
        "jp2": PlotProperty("JPEG2000", "limegreen", linewidth, alpha),
        "ours": PlotProperty("Ours", "purple", linewidth, alpha),
    }

    for method in plot_dict:
        df = (
            traditional_rate_distortion(img, cfg.img.name, extension=method)
            if method != "ours"
            else ours_df
        )
        print(method, df)

        plt.plot(
            df["Bytes"] / 1024,
            df["PSNR"],
            color=plot_dict[method].color,
            label=plot_dict[method].name,
            marker="o",
            linewidth=plot_dict[method].linewidth,
            alpha=plot_dict[method].alpha,
        )

    plt.legend(loc="lower right")
    plt.xlabel("Size (KB)")
    plt.ylabel("PSNR (dB)")
    # plt.title("Rate Distortion Curve")
    plt.grid()

    # Save to CSV
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
    path.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(path / f"{cfg.img.name}_rate_distortion.pdf", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
