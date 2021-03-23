"""
Run as:

python vis_tools/sparsify.py wandb.project=sparsify
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

    List all possible choices for (masking, init, density, dataset).

    We'll try matching the exhaustive caretesian product of
    (masking_ll x init_ll x suffix_ll x density_ll etc).

    :param runs: Experiment run
    :param masking_ll: List of sparse training techniques
    :param density_ll: List of density values (1 - sparsity)
    :param image_ll: List of images
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

        tags = [image, masking, str(density), suffix]

        # Remove tags that are None
        name = "_".join([tag for tag in tags if tag])
        logging.debug(name)

        run = run_dict.get(name, None)
        if not run:
            logging.info(f"Run {name} not present")
            continue

        psnr = run.summary.get("test_PSNR")

        df.loc[e] = [image, masking, density, psnr]

    df = df.sort_values(by=["Image", "Method", "Density"])

    if reorder:
        df = df.reset_index(drop=True)

    return df


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    # Get runs
    api = wandb.Api()
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")

    df = get_stats_table(
        runs,
        image_ll=["flower_foveon", "bridge", "big_building"],
        masking_ll=["RigL", "Small_Dense", "Pruning", "Feathermap"],
        suffix_ll=["train_5x"],
        density_ll=[0.05, 0.1, 0.2, 0.5, 0.75],
    )

    # Set longer length
    pd.options.display.max_rows = 150

    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    # Save to CSV
    path = Path(f"{hydra.utils.get_original_cwd()}/outputs/csv/")
    path.mkdir(exist_ok=True, parents=True)
    df.to_csv(path / f"sparsify_results.csv")


if __name__ == "__main__":
    main()
