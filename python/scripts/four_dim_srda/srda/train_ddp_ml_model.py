import argparse
import copy
import datetime
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.four_dim_srda.config.config_loader import load_config
from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import make_dataloaders_and_samplers
from src.four_dim_srda.models.loss_maker import make_loss
from src.four_dim_srda.models.model_maker import make_model
from src.four_dim_srda.models.optim_helper import optimize_ddp
from src.four_dim_srda.utils.log_maker import output_gpu_memory_summary_log
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=False)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--world_size", type=int, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--checkpoint_interval", type=int, default=10)
parser.add_argument("--override_num_epochs", type=int, default=None)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


def setup(rank: int, world_size: int, backend: str = "nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def save_training_checkpoint(
    *,
    rank: int,
    model: DDP,
    optimizer: Optimizer,
    checkpoint_path: str,
    epoch: int,
    es_cnt: int,
    best_epoch: int,
    best_loss: float,
    use_zero: bool,
):
    dist.barrier()
    if use_zero:
        optimizer.consolidate_state_dict(to=0)
    dist.barrier()

    if rank == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "es_cnt": es_cnt,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
        }
        tmp_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        logger.info(f"Checkpoint is saved at epoch {epoch + 1}: {checkpoint_path}")

    dist.barrier()


def train_and_validate(
    rank: int,
    world_size: int,
    config: BaseExperimentConfig,
    model_data_path: str,
    log_dir_path: str,
    checkpoint_interval: int,
):
    setup(rank, world_size)
    set_seeds(config.train_config.seed, use_deterministic=False)

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make dataloaders and samplers")
        logger.info("################################\n")

    dataloaders, samplers = make_dataloaders_and_samplers(
        cfg_dataloader=config.dataloader_config,
        cfg_data=config.dataset_config,
        root_dir=ROOT_DIR,
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make model, loss_fn, and optimizer")
        logger.info("###############################\n")

    model = make_model(cfg=config)
    model = DDP(model.to(rank), device_ids=[rank])
    loss_fn = make_loss(cfg=config.loss_config)

    if config.train_config.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=config.train_config.lr,
        )
        logger.info("ZeRO is used.")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train_config.lr)

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Train model")
        logger.info("###############################\n")

    best_weights_path = f"{model_data_path}/best_weights.pth"
    checkpoint_path = f"{model_data_path}/checkpoint.pth"
    learning_history_path = f"{log_dir_path}/learning_history.csv"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}")

        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info("Existing checkpoint weights & optimizer are loaded")

        es_cnt = checkpoint["es_cnt"]

        logger.info(f"early stopping count = {es_cnt}")

        df = pd.read_csv(learning_history_path)
        all_histories = df.to_dict(orient="records")
        epoch_start = checkpoint.get("epoch", len(all_histories) - 1) + 1
        all_histories = all_histories[:epoch_start]
        best_epoch = checkpoint.get("best_epoch", df["valid"].idxmin() + 1)
        best_loss = checkpoint.get("best_loss", df["valid"].min())

        logger.info("Learning history is loaded.")
        logger.info(f"epoch start: {epoch_start + 1}")
        logger.info(f"best epoch: {best_epoch}, and the loss: {best_loss}")

        # Remove checkpoint from memory
        del checkpoint
        # Clear the GPU cache
        torch.cuda.empty_cache()

    else:
        logger.info("No existing weights found, starting with random initialization")

        all_histories = []
        epoch_start = 0
        best_epoch = 0
        best_loss = np.inf

        logger.info("Learning history is initialized.")

    last_completed_epoch = epoch_start - 1
    for epoch in range(epoch_start, config.train_config.num_epochs):
        _time = time.time()
        histories = {}

        if rank == 0:
            logger.info(f"Epoch: {epoch + 1} / {config.train_config.num_epochs}")

        for mode in ["train", "valid"]:
            dataloaders[mode].dataset.set_hr_file_paths_randomly(epoch=epoch)
            dist.barrier()
            loss = optimize_ddp(
                dataloader=dataloaders[mode],
                sampler=samplers[mode],
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
                mode=mode,
            )
            histories[mode] = loss
            dist.barrier()

        elapsed_time_per_epoch = time.time() - _time
        histories["elapsed_time_per_epoch"] = elapsed_time_per_epoch

        all_histories.append(histories)
        last_completed_epoch = epoch

        if histories["valid"] > best_loss:
            es_cnt += 1
            if rank == 0:
                logger.info(f"ES count = {es_cnt}")
            if es_cnt >= config.train_config.early_stopping_patience:
                break
        else:
            best_epoch = epoch + 1
            best_loss = histories["valid"]
            es_cnt = 0

            if rank == 0:
                best_weights = copy.deepcopy(model.module.state_dict())
                torch.save(best_weights, best_weights_path)
                logger.info(
                    "Best loss is updated, ES count is reset, and model weights are saved."
                )

        if rank == 0:
            pd.DataFrame(all_histories).to_csv(learning_history_path, index=False)
            logger.info(f"Train loss = {histories['train']:.8f}")
            logger.info(f"Valid loss = {histories['valid']:.8f}")
            logger.info(f"Elapsed time = {histories['elapsed_time_per_epoch']} sec")
            logger.info("-----")

        if (epoch + 1) % checkpoint_interval == 0:
            save_training_checkpoint(
                rank=rank,
                model=model,
                optimizer=optimizer,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                es_cnt=es_cnt,
                best_epoch=best_epoch,
                best_loss=best_loss,
                use_zero=config.train_config.use_zero,
            )

    if rank == 0:
        pd.DataFrame(all_histories).to_csv(learning_history_path, index=False)

    if last_completed_epoch >= epoch_start:
        save_training_checkpoint(
            rank=rank,
            model=model,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            epoch=last_completed_epoch,
            es_cnt=es_cnt,
            best_epoch=best_epoch,
            best_loss=best_loss,
            use_zero=config.train_config.use_zero,
        )

    if rank == 0:
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}\n")

        #
        output_gpu_memory_summary_log()

    cleanup()


if __name__ == "__main__":
    try:
        os.environ["MASTER_ADDR"] = "localhost"

        # Port is arbitrary, but set random value to avoid collision
        np.random.seed(datetime.datetime.now().microsecond)
        port = str(np.random.randint(12000, 65535))
        os.environ["MASTER_PORT"] = port

        args = parser.parse_args()
        config_path = args.config_path
        world_size = args.world_size
        model_name = args.model_name
        checkpoint_interval = args.checkpoint_interval
        if checkpoint_interval <= 0:
            raise ValueError("--checkpoint_interval must be positive.")

        config = load_config(model_name=model_name, config_path=config_path)
        if args.override_num_epochs is not None:
            config.train_config.num_epochs = args.override_num_epochs

        experiment_name = config_path.split("/")[-4]
        config_name = os.path.basename(config_path).split(".")[0]

        model_data_path = f"{ROOT_DIR}/data/four_dim_srda/{experiment_name}/training/{model_name}/{config_name}"
        log_dir_path = f"{ROOT_DIR}/python/logs/four_dim_srda/{experiment_name}/training/{model_name}/{config_name}"
        os.makedirs(model_data_path, exist_ok=True)  # 後でFalseに変更
        os.makedirs(log_dir_path, exist_ok=True)  # 後でFalseに変更

        logger.addHandler(FileHandler(f"{log_dir_path}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        if not torch.cuda.is_available():
            logger.error("No GPU.")
            raise Exception("No GPU.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")
        logger.info(f"World size = {world_size}\n")
        logger.info(f"Checkpoint interval = {checkpoint_interval}\n")

        logger.info(f"config name = {config_name}")
        logger.info(f"Input config = {config.to_json_str()}")

        start_time = time.time()

        mp.spawn(
            train_and_validate,
            args=(world_size, config, model_data_path, log_dir_path, checkpoint_interval),
            nprocs=world_size,
            join=True,
        )

        end_time = time.time()

        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        logger.info("\n*********************************************************")
        logger.info(f"End DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
