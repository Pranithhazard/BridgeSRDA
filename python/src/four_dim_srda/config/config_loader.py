import pathlib
from src.four_dim_srda.config.config_u_net_maxvit_v1 import UNetMaxVitVer01ExpConfig
from src.four_dim_srda.config.experiment_config import BaseExperimentConfig


def load_config(model_name: str, config_path: str) -> BaseExperimentConfig:
    if model_name != "UNetMaxVitVer01":
        raise ValueError(f"Unsupported 4D-SRDA model: {model_name}")
    return UNetMaxVitVer01ExpConfig.load(pathlib.Path(config_path))
