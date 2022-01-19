from typing import List, Optional, Union
import yaml
from pathlib import Path
import shutil

from .config.settings import settings 

Pathlike = Union[str,Path]

def generate_configs(datasets:Optional[Union[List[str],str]]=None, 
                    checkpoint_dir:Optional[Pathlike]=None, 
                    datapath:Optional[Pathlike]=None,
                    config_dir:Optional[Pathlike]=None)->None:
    """Generate configuration files for training NDT models

    This dynamically generates config file(s) used for training NDT models.  
    It does this by setting correct paths pointing towards data files 
    and checkpoint files.  The paths for checkpoints and datafiles are given
    default values in config.settings and they can be set from the corresponding
    environment variables.  

    See Also:
        config.settings


    Args:
        datasets (Union[List[str], str]): The dataset or datasets to generate a 
            config for.  Can be a list of strings, each corresponding to 
            a dataset, or string corresponding to a single dataset.
            If None, generate all configs, default is None.
        checkpoint_dir (Union[Path, str]): The directory to save checkpoints
            into.  If None this will default to the value in 
            `config.settings.CHECKPOINT_DIR` which can be set with the 
            environment variable CHECKPOINT_DIR.
        datapath (Union[Path,str]): The path to the NDT training and validation 
            datasets.  If None this will default to the value in 
            `config.settings.NDT_CACHE_DIR` which can be set with the 
             environment variable NDT_CACHE_DIR
        config_dir (Union[Path,str]): The path to save the generated configs
            into.  If None this will default to the value in
            `config.settings.CONFIG_DIR` which can be set with the 
            environment variable CONFIG_DIR.
    """
    all_ = [
        'mc_maze',
        'mc_maze_small',
        'mc_maze_medium',
        'mc_maze_large',
        'mc_rtt',
        'area2_bump',
        'dmfc_rsg'
    ]

    if datasets is None:
        datasets = all_
    elif isinstance(datasets, str):
        datasets = [datasets]
    elif isinstance(datasets, list):
        for ds in datasets:
            if ds not in all_:
                raise ValueError(f"Dataset {ds} not a valid dataset name")
    
    config_dir = Path(config_dir or settings.CONFIG_DIR)

    for ds in datasets:
        with open(f'config_templates/{ds}.yaml', 'r') as f:
            config = yaml.safe_load(f.read())
        config['CHECKPOINT_DIR'] = str(checkpoint_dir or settings.CHECKPOINT_DIR)
        config['DATA']['DATAPATH'] = str(datapath or settings.NDT_CACHE_DIR)
        config['TRAIN']['TUNE_HP_JSON'] = str((Path(config_dir) / f"{ds}.json").resolve())

        Path(config_dir).mkdir(parents=True, exist_ok=True)
        with open(config_dir / f"{ds}.yaml", 'w') as f:
            f.write(yaml.dump(config))

        shutil.copy2(f"config_templates/{ds}.json", config_dir / f"{ds}.json")




