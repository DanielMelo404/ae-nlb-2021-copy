from pathlib import Path

from pydantic import BaseSettings


class HomeSettings(BaseSettings):
    NLB_HOME: Path = Path().home() / 'nlb_2021'


class Settings(BaseSettings):
    NLB_HOME = HomeSettings().NLB_HOME
    NLB_DATA_RAW: Path = NLB_HOME / 'raw'
    NLB_CACHE_DIR: Path = NLB_HOME / 'processed'
    NDT_CACHE_DIR: Path = NLB_HOME / 'neural-data-transformer'
    RAY_TUNE_HOME: Path = NLB_HOME / 'ray_results/neural-data-transformer'
    CHECKPOINT_DIR: Path = NLB_HOME / 'checkpoints'
    CONFIG_DIR: Path = NLB_HOME / 'configs'
    SUBMISSION_VALIDATION_ROOT: Path = NLB_HOME / 'submission-validation'


settings = Settings()
