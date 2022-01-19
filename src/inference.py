from typing import Dict, Optional

import torch
import numpy as np

from src.dataset import DATASET_MODES, SpikesDataset
from src.runner import Runner


def init_by_ckpt(ckpt_path:str, *, data_file:Optional[str]=None):
    """Initialize a runner from a checkpoint

    Adapted from https://github.com/snel-repo/neural-data-transformers/tree/master/scripts/analyze_utils.py

    This is neccesary to give us a hook into the setup before we load the data, 
    in particular, we need to change the runners path to the dataset so we 
    can change from the training data to the testing data for evaluation


    Args:
        ckpt_path (str): The path to the NDT checkpoint file.
        data_file (Optional[str]): The h5 file to use as input for model 
        predictions.  If None, use the training file provided 
        during model training.  Default is None.

    Returns:
        (Runner, np.ndarray, np.ndarray, np.ndarray, np.ndarray), The 
            runner, spikes, rates, heldout_spikes, and forward_spikes

    """
    runner = Runner(checkpoint_path=ckpt_path)
    if data_file is not None:
        # this is where we hack the file name
        runner.config.DATA.TRAIN_FILENAME = str(data_file)

    runner.model.eval()
    torch.set_grad_enabled(False)
    spikes, rates, heldout_spikes, forward_spikes = setup_dataset(runner)
    return runner, spikes, rates, heldout_spikes, forward_spikes
    
    
def get_output_rates(ckpt_path:str, dataset:str, data_file:Optional[str]=None)->Dict[str,np.ndarray]:
    """Compute model predictions from a checkpoint
    
    Adapted from https://github.com/snel-repo/neural-data-transformers/tree/master/scripts/nlb.py

    Args:
        ckpt_path (str): The path to the NDT checkpoint file.
        data_file (Optional[str]): The h5 file to use as input for model 
            predictions.  If None, use the training file provided 
            during model training.  Default is None.

    Returns:
        Dict[str,np.ndarray], The predictions from the model.
    """
    runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, data_file=data_file)

    eval_rates, _ = runner.get_rates(
        checkpoint_path=ckpt_path,
        save_path = None,
        mode = DATASET_MODES.val
    )
    train_rates, _ = runner.get_rates(
        checkpoint_path=ckpt_path,
        save_path = None,
        mode = DATASET_MODES.train
    )


    eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
    eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
    train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
    eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
    train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)



    output_dict = {
        dataset: {
            'train_rates_heldin': train_rates_heldin.cpu().numpy(),
            'train_rates_heldout': train_rates_heldout.cpu().numpy(),
            'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
            'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
            'eval_rates_heldin_forward': eval_rates_heldin_forward.cpu().numpy(),
            'eval_rates_heldout_forward': eval_rates_heldout_forward.cpu().numpy()
    }
        }

    return output_dict


def setup_dataset(runner:Runner):
    """Setup a dataset for evaluation
    
    Adapted from https://github.com/snel-repo/neural-data-transformers/tree/master/scripts/analyze_utils.py

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray), The spikes, rates,
            heldout_spikes, forward_spikes.    
    """
    test_set = SpikesDataset(runner.config, runner.config.DATA.TRAIN_FILENAME, logger=runner.logger)
    runner.logger.info(f"Evaluating on {len(test_set)} samples.")
    test_set.clip_spikes(runner.max_spikes)
    spikes, rates, heldout_spikes, forward_spikes = test_set.get_dataset()
    if heldout_spikes is not None:
        heldout_spikes = heldout_spikes.to(runner.device)
    if forward_spikes is not None:
        forward_spikes = forward_spikes.to(runner.device)
    return spikes.to(runner.device), rates.to(runner.device), heldout_spikes, forward_spikes
