"""
Run as:

python vis_tools/weight_removal.py wandb.project=sparsify
"""
import itertools
import logging
import os
from typing import List
from pathlib import Path
import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig
from matplotlib import pyplot as plt

COLORS = {
    "Small_Dense": "green",
    "Feathermap": "blue",
    "RigL": "purple",
    "Pruning": "brown",
    "siren": "black",
}
linewidth = 3
alpha = 0.8

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
    masking_ll: List[str] = ["RigL"],
    density_ll: List[float] = [0.1],
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
    columns = ["Image", "Method", "Density", "PSNR"]
    df = pd.DataFrame(columns=columns)

    # Pre-process
    logging.info("Grouping runs by name")
    run_dict = {}
    for run in runs:
        run_dict[run.name] = run

    for e, (image, masking, density, suffix) in enumerate(
        itertools.product(image_ll, masking_ll, density_ll, suffix_ll)
    ):

        tags = [image, masking, density, suffix]

        # Remove tags that are None
        name = "_".join([str(tag) for tag in tags if tag])
        logging.debug(name)

        run = run_dict.get(name, None)
        if not run:
            logging.debug(f"Run {name} not present")
            continue

        psnr = run.summary.get("test_PSNR")

        df.loc[e] = [image, masking, density, psnr]

    df = df.sort_values(by=["Image", "Method", "Density"])

    if reorder:
        df = df.reset_index(drop=True)

    return df


def sparsity_plot(
    df: pd.DataFrame,
    image_ll: List[str] = ["flower_foveon"],
    masking_ll: List[str] = ["RigL"],
):
    for image in image_ll:
        for masking in masking_ll:
            sub_df = df.loc[(df["Method"] == masking) & (df["Image"] == image)]
            color = COLORS[masking]

            if masking == "siren":
                label = "Baseline"
                plt.axhline(
                    y=sub_df["PSNR"].item(),
                    color=color,
                    label=label,
                    linewidth=linewidth,
                    alpha=alpha,
                )
            else:
                label = masking.replace("_", "-").replace("siren", "Baseline")
                plt.plot(
                    1 - sub_df["Density"],
                    sub_df["PSNR"],
                    color=color,
                    label=label,
                    marker="o",
                    linewidth=linewidth,
                    alpha=alpha,
                )

        xticks = [0.25, 0.5, 0.8, 0.9, 0.95]
        plt.xticks(xticks, [f"{(1 - xtick) * 100:.0f}" for xtick in xticks])
        plt.xlabel("% of Original Weights")
        plt.ylabel("PSNR (in dB)")
        plt.ylim(13, 45)
        plt.legend(loc="lower left")
        plt.grid()
        plt.tight_layout()

        # Save plot
        path = Path(f"{hydra.utils.get_original_cwd()}/outputs/plots/")
        path.mkdir(exist_ok=True, parents=True)

        plt.savefig(path / f"weight_removal_{image}.pdf", dpi=150)
        plt.show()


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    # Get runs
    api = wandb.Api()
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")

    image_ll = ["flower_foveon", "bridge", "big_building"]
    masking_ll = ["RigL", "Small_Dense", "Pruning", "Feathermap", "siren"]
    suffix_ll = ["train_5x"]
    density_ll = [0.05, 0.1, 0.2, 0.5, 0.75, None]

    df = get_stats_table(
        runs,
        image_ll=image_ll,
        masking_ll=masking_ll,
        density_ll=density_ll,
        suffix_ll=suffix_ll,
    )

    # Set longer length
    pd.options.display.max_rows = 150

    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    # Save to CSV
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/csv/")
    path.mkdir(exist_ok=True, parents=True)
    df.to_csv(path / f"sparsify_results.csv")

    sparsity_plot(df, image_ll, masking_ll)


if __name__ == "__main__":
    main()
