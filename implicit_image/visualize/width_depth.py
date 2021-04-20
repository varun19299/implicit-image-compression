"""
Run as:

python visualize/weight_removal.py wandb.project=masking
"""
import itertools
import logging
import os
from typing import List
from pathlib import Path
import hydra
import pandas as pd
import numpy as np
import wandb
from omegaconf import DictConfig
from matplotlib import pyplot as plt

COLORS = {
    "flower_foveon": "C0",
    "big_building": "C1",
    "bridge": "C2",
}
MARKERS = {"siren": "o", "fourier": "^"}
LINESTYLES = {"siren": "-", "fourier": "--"}

linewidth = 2

markersize = 8
alpha = 1.0

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


def get_stats_table(
    runs,
    image_ll: List[str] = ["flower_foveon"],
    model_ll: List[str] = ["RigL"],
    width_ll: List[int] = [128],
    depth_ll: List[int] = [128],
    suffix_ll: List[str] = ["train_5x"],
    reorder: bool = True,
) -> pd.DataFrame:
    """
    Get stats saved on W&B.

    List all possible choices for (image, masking, density, suffix).

    We'll try matching the exhaustive caretesian product of
    (masking_ll x init_ll x suffix_ll x density_ll etc).

    :param runs: Experiment run
    :param masking_ll: List of sparse training techniques
    :param density_ll: List of density values (1 - sparsity)
    :param image_ll: List of images
    :param suffix_ll: List of suffixes
    :param reorder: sort methods alphabetically

    :return: Dataframe containing test accuracies of methods
    :rtype: pd.DataFrame
    """
    columns = ["Image", "Model", "Width", "Depth", "PSNR"]
    df = pd.DataFrame(columns=columns)

    # Pre-process
    logging.info("Grouping runs by name")
    run_dict = {}
    for run in runs:
        run_dict[run.name] = run

    for e, (image, model, width, depth, suffix) in enumerate(
        itertools.product(image_ll, model_ll, width_ll, depth_ll, suffix_ll)
    ):

        tags = [image, model, f"w_{width}", f"d_{depth}", suffix]

        # Remove tags that are None
        name = "_".join([str(tag) for tag in tags if tag])
        logging.debug(name)

        run = run_dict.get(name, None)
        if not run:
            logging.debug(f"Run {name} not present")
            continue

        psnr = run.summary.get("test_PSNR")

        df.loc[e] = [image, model, width, depth, psnr]

    df = df.sort_values(by=["Image", "Model", "Width", "Depth"])

    if reorder:
        df = df.reset_index(drop=True)

    return df


def width_depth_plot(
    df: pd.DataFrame,
    image_ll: List[str] = ["flower_foveon"],
    model_ll: List[str] = ["siren"],
    width_constant: int = 128,
    depth_constant: int = 8,
):

    # Constant width
    depth_ll = None
    plt.figure(figsize=(7, 5))

    f = lambda m, c, ls: plt.plot(
        [], [], marker=m, color=c, ls=ls, linewidth=linewidth
    )[0]

    handles = [f(None, color, "-") for color in COLORS.values()]
    # handles += [f(marker, "k", "none") for marker in MARKERS.values()]
    handles += [f(None, "k", ls) for ls in LINESTYLES.values()]
    labels = list(COLORS.keys()) + list(LINESTYLES.keys())

    for image in image_ll:
        for model in model_ll:
            sub_df = df.loc[
                (df["Width"] == width_constant)
                & (df["Image"] == image)
                & (df["Model"] == model)
            ]
            color = COLORS[image]
            # marker = MARKERS[model]
            ls = LINESTYLES[model]

            plt.plot(
                sub_df["Depth"],
                sub_df["PSNR"],
                color=color,
                marker="o",
                markersize=markersize,
                ls=ls,
                linewidth=linewidth,
                alpha=alpha,
            )

            if not depth_ll:
                depth_ll = sub_df["Depth"].tolist()

    plt.xticks(depth_ll)
    plt.xlabel("Depth")
    plt.ylabel("PSNR (in dB)")
    plt.legend(handles, labels, loc="upper left")
    plt.grid()
    plt.tight_layout()

    # Save plot
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
    path.mkdir(exist_ok=True, parents=True)

    plt.savefig(path / "depth_variation.pdf", dpi=150)
    plt.show()

    # Constant depth
    width_ll = []
    plt.figure(figsize=(7, 5))
    for image in image_ll:
        for model in model_ll:
            sub_df = df.loc[
                (df["Depth"] == depth_constant)
                & (df["Image"] == image)
                & (df["Model"] == model)
            ]
            color = COLORS[image]
            # marker = MARKERS[model]
            ls = LINESTYLES[model]

            plt.plot(
                sub_df["Width"] ** 0.5,
                sub_df["PSNR"],
                color=color,
                marker="o",
                markersize=markersize,
                ls=ls,
                linewidth=linewidth,
                alpha=alpha,
            )

            if not len(width_ll):
                width_ll = sub_df["Width"]

    plt.xticks(width_ll ** 0.5, width_ll)
    plt.xlabel("Width")
    plt.ylabel("PSNR (in dB)")
    plt.legend(handles, labels, loc="upper left")
    plt.grid()
    plt.tight_layout()

    # Save plot
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
    path.mkdir(exist_ok=True, parents=True)

    plt.savefig(path / "width_variation.pdf", dpi=150)
    plt.show()


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    # Get runs
    api = wandb.Api()
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")

    image_ll = ["flower_foveon", "bridge", "big_building"]
    model_ll = ["siren", "fourier"]
    suffix_ll = ["train_5x"]
    width_ll = [64, 96, 128, 256]
    depth_ll = [4, 6, 8, 10]

    df = get_stats_table(
        runs,
        image_ll=image_ll,
        model_ll=model_ll,
        width_ll=width_ll,
        depth_ll=depth_ll,
        suffix_ll=suffix_ll,
    )

    # Set longer length
    pd.options.display.max_rows = 150

    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    # Save to CSV
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/csv/")
    path.mkdir(exist_ok=True, parents=True)
    df.to_csv(path / f"width-depth_results.csv")

    width_depth_plot(df, image_ll, model_ll)


if __name__ == "__main__":
    main()
