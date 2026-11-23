from logging import getLogger
from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.models.neural_nets.u_net_maxvit_v1 import UNetMaxVitVer01
from torch import nn


logger = getLogger()


def make_model(cfg: BaseExperimentConfig) -> nn.Module:
    if cfg.model_config.model_name != "UNetMaxVitVer01":
        raise NotImplementedError(f"{cfg.model_config.model_name} is not supported.")
    logger.info("UNetMaxVitVer01 is created.")
    return UNetMaxVitVer01(cfg.model_config)
