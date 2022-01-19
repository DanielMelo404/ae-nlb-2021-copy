import logging
import subprocess

import click

from src.config.settings import settings
from src.data import NDTDataHandler

logger = logging.getLogger(__name__)


@click.command()
@click.argument("dataset_name")
@click.option("--samples", type=int, default=100, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def cli(dataset_name: str, samples: int, seed: int):
    """CLI Interface for running hyperparameter optimization with NDT

    Args:
        dataset_name (str): The name of the NLB dataset to run
            NDT hyperoptimzation against
        samples (int, optional): The number of model samples to train. Defaults to 100.
        seed (int, optional): The random seed to use when beginning the hyperparameter optimization. Defaults to 42.
    """
    run_hyperopt(dataset_name, samples, seed)


def run_hyperopt(dataset_name: str, samples: int = 100, seed: int = 42):
    """Run a hyperoptimization against a single dataset

    Args:
        dataset (str): The name of the NLB dataset to run
            NDT hyperoptimization against
        seed (int, optional): The random seed to use when beginning the hyperparameter optimization. Defaults to 42.
    """
    NDTDataHandler().make_train_data(dataset_name)

    script_path = str("ray_bayesopt.py")
    script_flag = "--exp-config"
    script_arg = str(settings.CONFIG_DIR / f"{dataset_name}.yaml")
    additional_args = [
        "--samples", str(samples),
        "--gpus-per-worker", "0.5",
        "--cpus-per-worker", "3",
        "--seed", str(seed)
    ]
    run_args = ["python", script_path, script_flag, script_arg, *additional_args]
    logger.info(f"Running command {' '.join(run_args)}")
    subprocess.run(run_args)


if __name__ == "__main__":
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    cli()
