import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].resolve()))

import yaml
import pytest
from src.generate_configs import settings
from src.generate_configs import generate_configs


@pytest.fixture
def config_dir(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def explict_test_config(monkeypatch, config_dir):
    """Set explict default configs for testing"""
    monkeypatch.setattr(settings, 'CHECKPOINT_DIR', Path('fake_ckpt_dir'))
    monkeypatch.setattr(settings, 'NDT_CACHE_DIR', Path('fake_ndt_dir'))
    monkeypatch.setattr(settings, 'CONFIG_DIR', config_dir)
    



def test_generate_configs_writes_single_dataset(config_dir):
    """Verify generate_configs correctly writes config for a single dataset"""
    test_cases = [
        ('mc_maze', config_dir / 'mc_maze.json'),
        ('mc_maze_small', config_dir/ 'mc_maze_small.json'),
        ('mc_maze_medium', config_dir / 'mc_maze_medium.json'),
        ('mc_maze_large', config_dir / 'mc_maze_large.json'),
        ('mc_rtt', config_dir / 'mc_rtt.json'),
        ('area2_bump', config_dir / 'area2_bump.json'),
        ('dmfc_rsg', config_dir / 'dmfc_rsg.json'),
    ]

    for dataset, tune_file in test_cases:

        # write the config
        generate_configs(dataset)

        # assert written correctly
        with open(settings.CONFIG_DIR / f"{dataset}.yaml", 'r') as f:
            result = yaml.safe_load(f)
            assert result['CHECKPOINT_DIR'] == 'fake_ckpt_dir'
            assert result['DATA']['DATAPATH'] == 'fake_ndt_dir'
            assert result['TRAIN']['TUNE_HP_JSON'] == str(tune_file)
            


def test_generate_configs_writes_all_datasets(config_dir):
    """Verify that when no arguments are given all dataset configs are written"""
    test_cases = [
        ('mc_maze', config_dir / 'mc_maze.json'),
        ('mc_maze_small', config_dir/ 'mc_maze_small.json'),
        ('mc_maze_medium', config_dir / 'mc_maze_medium.json'),
        ('mc_maze_large', config_dir / 'mc_maze_large.json'),
        ('mc_rtt', config_dir / 'mc_rtt.json'),
        ('area2_bump', config_dir / 'area2_bump.json'),
        ('dmfc_rsg', config_dir / 'dmfc_rsg.json'),
    ]

    # write all configs - default when no arguments are given
    generate_configs()
    for dataset, tune_file in test_cases:

        # assert written correctly
        with open(settings.CONFIG_DIR / f"{dataset}.yaml", 'r') as f:
            result = yaml.safe_load(f)
            assert result['CHECKPOINT_DIR'] == 'fake_ckpt_dir'
            assert result['DATA']['DATAPATH'] == 'fake_ndt_dir'
            assert result['TRAIN']['TUNE_HP_JSON'] == str(tune_file) 

def test_generate_configs_writes_subset_of_datasets(config_dir):
    """Verify that when a subset of datasets are given then only those are written"""
    datasets = ['mc_maze', 'area2_bump', 'dmfc_rsg']
    tune_files = [config_dir / f"{ds}.json" for ds in datasets]

    # write configs
    generate_configs(datasets)
    for dataset, tune_file in zip(datasets, tune_files):

        # assert written correctly
        with open(settings.CONFIG_DIR / f"{dataset}.yaml", 'r') as f:
            result = yaml.safe_load(f)
            assert result['CHECKPOINT_DIR'] == 'fake_ckpt_dir'
            assert result['DATA']['DATAPATH'] == 'fake_ndt_dir'
            assert result['TRAIN']['TUNE_HP_JSON'] == str(tune_file) 

