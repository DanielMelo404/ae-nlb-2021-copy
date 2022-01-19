from collections import defaultdict
import logging
import os
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd

from nlb_tools.make_tensors import h5_to_dict
from src.config.settings import settings
from src.data import NDTDataHandler
from src.inference import get_output_rates


logger = logging.getLogger(os.path.basename(__file__))


AESMTE3_ENSEMBLE_SIZES = {
    "area2_bump": 21,
    "dmfc_rsg": 13,
    "mc_maze": 8,
    "mc_maze_large": 8,
    "mc_maze_medium": 8,
    "mc_maze_small": 7,
    "mc_rtt": 13,
}


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("submission_file", type=Path)
def cli(dataset_name: str, submission_file: Path) -> None:
    """This script is intended to be used to validate AE Studio's AESMTE3
    submission to the NLB 2021 Challenge.
    
    Checkpoints should already be downloaded and saved to the path defined
    by environment variable SUBMISSION_VALIDATION_ROOT. Run
    `download_checkpoints.py` if you haven't already to get the checkpoints.

    \b
    DATASET_NAME is the name of the NLB dataset for which the submission
        should be validated.
    SUBMISSION_FILE is the path to the submission file which should be
        validated.
    """
    model_checkpoint_root = settings.SUBMISSION_VALIDATION_ROOT

    validate_arguments(dataset_name, submission_file, model_checkpoint_root)
    logger.info(f"Validating submissions for dataset: {dataset_name}")

    submission_rates = load_submission_file(submission_file)
    if dataset_name not in submission_rates:
        msg = f"Submission file does not contain rates for dataset {dataset_name}"
        logger.error(msg)
        raise KeyError(msg)

    logger.info("Selecting top models for use in ensemble")
    n_models = AESMTE3_ENSEMBLE_SIZES[dataset_name]
    models_df = get_top_model_metadata(dataset_name, n_models, model_checkpoint_root)
    model_output_rates = get_model_test_rate_outputs(
        dataset_name, models_df, model_checkpoint_root
    )

    logger.info("Ensembling models")
    ensemble_rates = ensemble_model_outputs(dataset_name, model_output_rates)

    logger.info("Comparing ensemble rates against the submission rates")
    try:
        compare_outputs(dataset_name, submission_rates, ensemble_rates)
    except AssertionError:
        logger.error("Validation failed! Tensors do not match.")
        raise
    else:
        logger.info("All tensors match, validation is a success!")


def validate_arguments(
    dataset_name: str, submission_file: Path, model_checkpoint_root: Path
) -> None:
    """Validate the argument inputs to the script.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        submission_file (Path): The path to the submission file to load.
        model_checkpoint_root (Path): The root directory of the extracted
            model checkpoints.

    Raises:
        ValueError: if dataset_name is invalid.
        FileNotFoundError: if either submission_file or model_checkpoint_root
            does not exist.
    """
    valid_datasets = [
        "mc_maze",
        "mc_rtt",
        "area2_bump",
        "dmfc_rsg",
        "mc_maze_small",
        "mc_maze_medium",
        "mc_maze_large",
    ]
    if dataset_name not in valid_datasets:
        msg = f"{str(dataset_name)} is not a valid dataset."
        logger.error(msg)
        raise ValueError(msg)

    if not submission_file.exists():
        msg = f"File {str(submission_file)} does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not model_checkpoint_root.exists():
        msg = f"Directory {str(model_checkpoint_root)} does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)


def load_submission_file(submission_file: Path) -> dict:
    """Loads an h5 EvalAI submission file into memory.

    The submission file should be an h5 file which matches the submission
    format described on the EvalAI submission page:
    https://eval.ai/web/challenges/challenge-page/1256/submission

    Args:
        submission_file (Path): The path to the submission file to load.

    Returns:
        dict: A nested dictionary with numpy arrays of firing rates matching
            the submission structure.
    """
    logger.info(f"Loading submission data from: {str(submission_file)}")
    with h5py.File(submission_file, "r") as f:
        return h5_to_dict(f)


def get_top_model_metadata(
    dataset_name: str, n_models: int, model_checkpoint_root: Path
) -> pd.DataFrame:
    """Returns a dataframe of metadata about the top models.

    Models are ordered (decreasing) by the validation co-bps in the 'co-bps'
    column of the model metadata, and the top n_models rows are returned.

    The returned metadata is intended to be used in order to ensemble the
    selected models together.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        n_models (int): The number of models to include in the ensemble.
        model_checkpoint_root (Path): The root directory of the extracted
            model checkpoints.

    Returns:
        pd.DataFrame: A dataframe containing metadata about the
            best-performing models which can be used for ensembling.
    """
    all_models = load_model_metadata(dataset_name, model_checkpoint_root)
    top_models = all_models.sort_values("co-bps", ascending=False).head(n_models)
    if dataset_name == "dmfc_rsg":
        # Our dmfc_rsg submission swapped out trial 'a6c8e99c' for '91fde75c'
        # so we need to do that swap here to get matching rates
        top_models.loc[top_models.trial_id == "a6c8e99c", "trial_id"] = "91fde75c"
    return top_models


def load_model_metadata(dataset_name: str, model_checkpoint_root: Path) -> pd.DataFrame:
    """Loads a model metadata csv into a pandas dataframe.

    The target file is the `models.csv` file located together with model
    checkpoints released alongside this codebase.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        model_checkpoint_root (Path): The root directory of the extracted
            model checkpoints.

    Returns:
        pd.DataFrame: A dataframe containing metadata about the models and
            their validation results.
    """
    metadata_path = model_checkpoint_root / dataset_name / "models.csv"
    logger.info(f"  Loading model metadata from: {str(metadata_path)}")
    return pd.read_csv(metadata_path, index_col=0)


def get_model_test_rate_outputs(
    dataset_name: str, models_df: pd.DataFrame, model_checkpoint_root: Path
) -> list:
    """Returns test rate outputs for all of the specified models.

    The model rates are returned as a list where each element is the
    dictionary structure containing output test rates for a single model.
    The order of the list corresponds to the order of the rows in models_df.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        models_df (pd.DataFrame): Metadata for the models to evaluate.
        model_checkpoint_root (Path): The root directory of the extracted
            model checkpoints.

    Returns:
        list: A list containing the inferred test rates for the models
            specified in models_df.
    """

    def _get_test_rates(row):
        return get_model_outputs(
            dataset_name, "test", model_checkpoint_root, row.trial_id, row.val_ckpt
        )
    # Ensure that the necessary ndt dataset has been pre-built
    NDTDataHandler().make_test_data(dataset_name)
    return [_get_test_rates(row) for row in models_df.itertuples()]


def get_model_outputs(
    dataset_name: str,
    phase: str,
    model_checkpoint_root: Path,
    trial_id: str,
    val_ckpt: str,
) -> dict:
    """Loads a model checkpoint and runs inference to get output rates.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        phase (str): The phase to evaluate. Must be one of: ['val', 'test']
        model_checkpoint_root (Path): The root directory of the extracted
            model checkpoints.
        trial_id (str): The trial id assigned to this model by ray tune.
        val_ckpt (str): Indicates which of the checkpoints for the trial
            should be loaded. Must be one of: ['lfve', 'lve']. 'lfve'
            checkpoint has the best unmasked loss, and 'lve' has the best
            masked loss.

    Returns:
        dict: A nested dictionary with numpy arrays of output rates matching
            the submission structure.
    """
    logger.info(f"  Inferring {phase} rates for model ({trial_id}, {val_ckpt})")
    ckpt_path = model_checkpoint_root / dataset_name / f"{trial_id}.{val_ckpt}.pth"
    ndt_file_path = NDTDataHandler().get_filename(dataset_name, phase=phase)
    return get_output_rates(ckpt_path, dataset_name, data_file=ndt_file_path)


def ensemble_model_outputs(dataset_name: str, model_outputs: list) -> dict:
    """Ensembles the outputs from multiple models together.

    Ensembling is done by taking the mean across models for each output
    tensor.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        model_outputs (list): The list of outputs from the models to ensemble.

    Returns:
        dict: A nested dictionary with numpy arrays of ensembled output rates.
    """
    reorganized_tensors = defaultdict(list)
    for tensor_dict in model_outputs:
        for tensor_name, rates in tensor_dict[dataset_name].items():
            reorganized_tensors[tensor_name].append(rates)
    return {
        dataset_name: {
            k: np.stack(v).mean(axis=0) for k, v in reorganized_tensors.items()
        }
    }


def compare_outputs(dataset_name: str, outputs_1: dict, outputs_2: dict) -> None:
    """Verifies that two model outputs contain the same tensors for the
    specified dataset.

    This evaluation uses the numpy assertion assert_allclose to verify that
    all output rate tensors match to a relative tolerance of 1e-5.

    Args:
        dataset_name (str): The name of the NLB dataset that the submission
            being validated targets.
        outputs_1 (dict): The nested dict containing the first set of output
            rates to compare.
        outputs_2 (dict): The nested dict containing the second set of output
            rates to compare.

    Raises:
        AssertionError: If the output rate tensors differ by more than a
            relative tolerance of 1e-5 in any element.
    """
    tensors = [
        "train_rates_heldin",
        "train_rates_heldout",
        "eval_rates_heldin",
        "eval_rates_heldout",
        "eval_rates_heldin_forward",
        "eval_rates_heldout_forward",
    ]
    for tensor in tensors:
        np.testing.assert_allclose(
            outputs_1[dataset_name][tensor], outputs_2[dataset_name][tensor], rtol=1e-5
        )


if __name__ == "__main__":
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    cli()
