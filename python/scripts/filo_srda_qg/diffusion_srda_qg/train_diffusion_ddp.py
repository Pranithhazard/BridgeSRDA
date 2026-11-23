from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import json
import os
import pathlib
import random
import socket
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from src.filo_srda_qg.diffusion_srda_qg.data import (
    diffusion_values_to_zero_one,
    make_diffusion_dataset,
    make_diffusion_loader,
)
from src.filo_srda_qg.diffusion_srda_qg.diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
)
from src.filo_srda_qg.diffusion_srda_qg.metrics import ensemble_summary
from src.filo_srda_qg.diffusion_srda_qg.model import make_diffusion_model
from src.filo_srda_qg.shared_srda.source import fold_pv
from src.filo_srda_qg.utils.checkpointing import (
    atomic_torch_save,
    load_model_state,
    model_state,
)
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--override-num-epochs", type=int)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--run-name")
    return parser.parse_args()


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve(root: pathlib.Path, path: str | pathlib.Path) -> pathlib.Path:
    value = pathlib.Path(path)
    return value if value.is_absolute() else root / value


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def reduce_mean(total: torch.Tensor, count: torch.Tensor) -> float:
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return (total / count.clamp_min(1)).item()


def write_failure_diagnostics(
    log_dir: pathlib.Path,
    *,
    rank: int,
    payload: dict[str, Any],
) -> None:
    path = log_dir / f"failure_rank{rank}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))


@torch.no_grad()
def validation_panel(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: Any,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    output_path: pathlib.Path,
) -> dict[str, float]:
    reset_rng(seed)
    count = min(int(config["train"]["validation_panel_size"]), len(dataset))
    samples = [dataset[index] for index in range(count)]
    target = fold_pv(torch.stack([sample["hr"] for sample in samples]))
    panel_generator = torch.Generator(device=device).manual_seed(seed + 1)
    panel_ensembles: list[torch.Tensor] = []
    sampling_seconds = 0.0
    panel_nfe = 0
    for sample in samples:
        result = diffusion.sample(
            model,
            condition_lr=sample["condition_lr"][None].to(device),
            obs=sample["obs"][None].to(device),
            ensemble_size=int(config["train"]["validation_panel_ensemble"]),
            steps=int(config["train"]["validation_panel_steps"]),
            eta=float(config["train"]["validation_panel_eta"]),
            chunk_size=int(config["train"]["validation_panel_chunk"]),
            generator=panel_generator,
        )
        if not torch.isfinite(result.samples).all():
            raise FloatingPointError("Validation-panel diffusion samples are non-finite")
        panel_ensembles.append(result.samples)
        sampling_seconds += result.elapsed_seconds
        panel_nfe = result.nfe
    panel_samples = torch.cat(panel_ensembles, dim=0)
    data_scaling = str(config["dataset"].get("data_scaling", "zero_one"))
    metric_samples = diffusion_values_to_zero_one(
        panel_samples, data_scaling=data_scaling
    )
    metric_target = diffusion_values_to_zero_one(
        target, data_scaling=data_scaling
    )
    reconstruction_l1 = (
        panel_samples - target[:, None]
    ).abs().mean().item()
    metrics = ensemble_summary(
        metric_samples,
        metric_target,
        pv_min=float(config["dataset"]["pv_min"]),
        pv_max=float(config["dataset"]["pv_max"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        target=metric_target.numpy().astype(np.float32),
        ensemble=metric_samples.numpy().astype(np.float32),
    )
    try:
        import matplotlib.pyplot as plt

        frame_index = min(4, target.shape[1] - 1)
        z_index = target.shape[2] // 2
        columns = 2 + min(metric_samples.shape[1], 4)
        figure, axes = plt.subplots(
            target.shape[0],
            columns,
            figsize=(3.2 * columns, 2.7 * target.shape[0]),
            squeeze=False,
        )
        for batch_index in range(target.shape[0]):
            panels = [
                ("target", metric_target[batch_index, frame_index, z_index]),
                (
                    "ensemble mean",
                    metric_samples[batch_index, :, frame_index, z_index].mean(dim=0),
                ),
            ]
            panels.extend(
                (
                    f"member {member}",
                    metric_samples[batch_index, member, frame_index, z_index],
                )
                for member in range(min(metric_samples.shape[1], 4))
            )
            vmin = metric_target[batch_index, frame_index, z_index].min().item()
            vmax = metric_target[batch_index, frame_index, z_index].max().item()
            for axis, (title, values) in zip(axes[batch_index], panels):
                axis.imshow(values, cmap="coolwarm", vmin=vmin, vmax=vmax)
                axis.set_title(title)
                axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path.with_suffix(".png"), dpi=140)
        plt.close(figure)
    except ImportError:
        print("warning: matplotlib unavailable; validation panel PNG was not written")
    metrics["sampling_seconds"] = sampling_seconds
    metrics["nfe"] = float(panel_nfe)
    metrics["reconstruction_l1"] = reconstruction_l1
    return {f"panel_{key}": value for key, value in metrics.items()}


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)
    requested_run_name = args.run_name or str(config["output"]["run_name"])
    config["output"]["run_name"] = requested_run_name

    root = pathlib.Path(args.root_dir).resolve()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if cuda else "gloo", rank=rank, world_size=world_size
        )
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    seed = int(config["train"]["seed"])
    set_seeds(seed + rank)

    work_root = resolve(root, config["output"]["work_root"])
    run_name = requested_run_name
    checkpoint_dir = work_root / "checkpoints" / run_name
    log_dir = work_root / "logs" / "training" / run_name
    panel_dir = work_root / "samples" / run_name / "validation_panel"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    best_weights_path = checkpoint_dir / "best_weights.pth"
    best_noise_weights_path = checkpoint_dir / "best_noise_weights.pth"
    best_reconstruction_weights_path = (
        checkpoint_dir / "best_reconstruction_weights.pth"
    )
    history_path = log_dir / "learning_history.csv"
    snapshot_path = checkpoint_dir / "config_snapshot.yml"

    checkpoint_data: dict[str, Any] | None = None
    if checkpoint_path.exists():
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        resolved = checkpoint_data.get("resolved_config")
        if isinstance(resolved, dict):
            config = copy.deepcopy(resolved)
            config["output"]["run_name"] = run_name
        width_key = (
            "base_width"
            if config["model"].get("model_name") == "DiffSRDAUNet3D"
            else "width"
        )
        if args.model_width is not None and int(args.model_width) != int(
            config["model"][width_key]
        ):
            raise ValueError(
                "Cannot change model width while resuming an existing checkpoint"
            )
        head_key = (
            "attention_heads"
            if config["model"].get("model_name") == "DiffSRDAUNet3D"
            else "n_head"
        )
        if args.n_head is not None and int(args.n_head) != int(
            config["model"][head_key]
        ):
            raise ValueError(
                "Cannot change attention heads while resuming an existing checkpoint"
            )

    if args.override_num_epochs is not None:
        config["train"]["num_epochs"] = int(args.override_num_epochs)
    if args.activation_checkpointing:
        config["model"]["activation_checkpointing"] = True
    if args.model_width is not None:
        width_key = (
            "base_width"
            if config["model"].get("model_name") == "DiffSRDAUNet3D"
            else "width"
        )
        config["model"][width_key] = int(args.model_width)
    if args.n_head is not None:
        head_key = (
            "attention_heads"
            if config["model"].get("model_name") == "DiffSRDAUNet3D"
            else "n_head"
        )
        config["model"][head_key] = int(args.n_head)

    train_dataset = make_diffusion_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="train",
        samples_per_epoch=int(config["train"]["samples_per_epoch"]),
    )
    valid_dataset = make_diffusion_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="valid",
        samples_per_epoch=int(config["train"]["valid_samples_per_epoch"]),
    )
    microbatch = int(config["train"]["microbatch_per_gpu"])
    train_loader, train_sampler = make_diffusion_loader(
        dataset=train_dataset,
        split="train",
        batch_size=microbatch,
        num_workers=int(config["dataloader"]["num_workers"]),
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    valid_loader, _ = make_diffusion_loader(
        dataset=valid_dataset,
        split="valid",
        batch_size=microbatch,
        num_workers=0,
        seed=seed + 10_000,
        rank=rank,
        world_size=world_size,
    )

    model = make_diffusion_model(config["model"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if cuda else None)
    diffusion = GaussianDiffusion(DiffusionConfig.from_dict(config["diffusion"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    amp_enabled = bool(config["train"]["amp"]) and cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_epoch = 0
    best_loss = float("inf")
    best_noise_epoch = 0
    best_noise_loss = float("inf")
    best_reconstruction_epoch = 0
    best_reconstruction_loss = float("inf")
    early_stopping_count = 0
    amp_overflow_skips = 0
    consecutive_amp_overflow_skips = 0
    history: list[dict[str, float]] = []
    if checkpoint_data is not None:
        load_model_state(model, checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        start_epoch = int(checkpoint_data["epoch"]) + 1
        best_epoch = int(checkpoint_data["best_epoch"])
        best_loss = float(checkpoint_data["best_loss"])
        best_noise_epoch = int(
            checkpoint_data.get("best_noise_epoch", best_epoch)
        )
        best_noise_loss = float(
            checkpoint_data.get("best_noise_loss", best_loss)
        )
        best_reconstruction_epoch = int(
            checkpoint_data.get("best_reconstruction_epoch", 0)
        )
        best_reconstruction_loss = float(
            checkpoint_data.get("best_reconstruction_loss", float("inf"))
        )
        early_stopping_count = int(checkpoint_data["early_stopping_count"])
        amp_overflow_skips = int(checkpoint_data.get("amp_overflow_skips", 0))
        consecutive_amp_overflow_skips = int(
            checkpoint_data.get("consecutive_amp_overflow_skips", 0)
        )
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")[:start_epoch]

    num_epochs = int(config["train"]["num_epochs"])
    patience = int(config["train"]["early_stopping_patience"])
    checkpoint_selection = str(
        config["train"].get("checkpoint_selection", "noise_l1")
    )
    reconstruction_selection_metric = str(
        config["train"].get(
            "reconstruction_selection_metric", "ensemble_mean_mae"
        )
    )
    if reconstruction_selection_metric not in {
        "ensemble_mean_mae",
        "sample_l1",
    }:
        raise ValueError(
            "Unsupported reconstruction selection metric: "
            f"{reconstruction_selection_metric}"
        )
    use_recon_loss = bool(config["train"].get("use_recon_loss", False))
    if use_recon_loss != (checkpoint_selection == "reconstruction_mae"):
        raise ValueError(
            "use_recon_loss and checkpoint_selection must agree: reconstruction "
            "selection requires use_recon_loss=true"
        )
    if checkpoint_selection not in {"noise_l1", "reconstruction_mae"}:
        raise ValueError(
            f"Unsupported checkpoint selection: {checkpoint_selection}"
        )
    if rank == 0:
        with snapshot_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        print(f"start={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        print(f"host={socket.gethostname()} run_name={run_name}")
        print(f"parameter_count={parameter_count:,} world_size={world_size}")
        print(
            f"microbatch={microbatch} accumulation={config['train']['gradient_accumulation_steps']} "
            f"effective_global_batch={microbatch * world_size * int(config['train']['gradient_accumulation_steps'])}"
        )
        print(
            f"amp={amp_enabled} checkpoint_selection={checkpoint_selection} "
            "reconstruction_selection_metric="
            f"{reconstruction_selection_metric} "
            f"validation_panel_interval={config['train']['validation_panel_interval']}"
        )
    if start_epoch >= num_epochs or early_stopping_count >= patience:
        if rank == 0:
            reason = (
                "maximum epoch reached"
                if start_epoch >= num_epochs
                else "early stopping already reached"
            )
            print(
                f"checkpoint already completed epoch {start_epoch}/{num_epochs}: "
                f"{reason}"
            )
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    accumulation = int(config["train"]["gradient_accumulation_steps"])
    validation_seed = seed + 100_000
    last_epoch = start_epoch - 1
    checkpoint_payload: dict[str, Any] = {}
    for epoch in range(start_epoch, num_epochs):
        epoch_started = time.time()
        epoch_seed = seed + 1_000_000 * epoch + rank
        reset_rng(epoch_seed)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(epoch_seed)
        train_dataset.set_epoch(epoch)
        valid_dataset.set_epoch(0)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = torch.tensor(0.0, device=device)
        train_count = torch.tensor(0.0, device=device)
        epoch_amp_overflow_skips = 0
        for batch_index, batch in enumerate(train_loader):
            condition_lr = batch["condition_lr"].to(device, non_blocking=True)
            obs = batch["obs"].to(device, non_blocking=True)
            target = fold_pv(batch["hr"].to(device, non_blocking=True))
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            sync = (
                contextlib.nullcontext()
                if should_step or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync:
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    raw_loss = diffusion.training_loss(
                        model,
                        target=target,
                        condition_lr=condition_lr,
                        obs=obs,
                    )
                    loss = raw_loss / float(group_size)
                if not torch.isfinite(raw_loss):
                    write_failure_diagnostics(
                        log_dir,
                        rank=rank,
                        payload={
                            "kind": "non_finite_training_loss",
                            "epoch": epoch,
                            "batch": batch_index,
                            "loss": raw_loss.detach().float().cpu().item(),
                        },
                    )
                    raise FloatingPointError(
                        f"Non-finite loss epoch={epoch} batch={batch_index}"
                    )
                scaler.scale(loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["train"]["gradient_clip_norm"])
                )
                gradient_is_finite = torch.isfinite(gradient_norm)
                overflow_flag = torch.tensor(
                    0 if gradient_is_finite else 1,
                    device=device,
                    dtype=torch.int32,
                )
                if dist.is_initialized():
                    dist.all_reduce(overflow_flag, op=dist.ReduceOp.MAX)
                if overflow_flag.item():
                    scaler_scale = float(scaler.get_scale())
                    write_failure_diagnostics(
                        log_dir,
                        rank=rank,
                        payload={
                            "kind": "non_finite_gradient_norm",
                            "epoch": epoch,
                            "batch": batch_index,
                            "gradient_norm": gradient_norm.detach()
                            .float()
                            .cpu()
                            .item(),
                            "amp_enabled": amp_enabled,
                            "scaler_scale": scaler_scale,
                            "recoverable_amp_overflow": amp_enabled,
                        },
                    )
                    max_overflow_skips = int(
                        config["train"].get(
                            "max_consecutive_amp_overflow_skips",
                            config["train"].get("max_amp_overflow_skips", 8),
                        )
                    )
                    if (
                        not amp_enabled
                        or consecutive_amp_overflow_skips >= max_overflow_skips
                    ):
                        raise FloatingPointError("Non-finite gradient norm")
                    amp_overflow_skips += 1
                    consecutive_amp_overflow_skips += 1
                    epoch_amp_overflow_skips += 1
                    new_scale = max(scaler_scale * 0.5, 1.0)
                    scaler.update(new_scale=new_scale)
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        print(
                            "warning: skipped optimizer step after synchronized "
                            f"AMP overflow epoch={epoch + 1} batch={batch_index} "
                            f"scale={scaler_scale:g}->{new_scale:g} "
                            f"consecutive={consecutive_amp_overflow_skips}/"
                            f"{max_overflow_skips} total={amp_overflow_skips}"
                        )
                    continue
                scaler.step(optimizer)
                scaler.update()
                consecutive_amp_overflow_skips = 0
                optimizer.zero_grad(set_to_none=True)
            train_total += raw_loss.detach() * target.shape[0]
            train_count += target.shape[0]
        row: dict[str, float] = {"train": reduce_mean(train_total, train_count)}
        row["amp_overflow_skips"] = float(epoch_amp_overflow_skips)

        reset_rng(validation_seed)
        valid_generator = torch.Generator(device=device).manual_seed(validation_seed)
        model.eval()
        # Evaluation shards are intentionally disjoint and may have different
        # lengths (including zero on small smoke datasets).  Forwarding through
        # DDP would run buffer-synchronization collectives only on ranks that
        # own validation samples, causing collective-order mismatches.  No
        # gradient synchronization is needed here, so evaluate the underlying
        # module and reduce only the scalar totals below.
        validation_model = model.module if hasattr(model, "module") else model
        valid_total = torch.tensor(0.0, device=device)
        valid_count = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for batch in valid_loader:
                condition_lr = batch["condition_lr"].to(device)
                obs = batch["obs"].to(device)
                target = fold_pv(batch["hr"].to(device))
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    loss = diffusion.training_loss(
                        validation_model,
                        target=target,
                        condition_lr=condition_lr,
                        obs=obs,
                        generator=valid_generator,
                    )
                if not torch.isfinite(loss):
                    write_failure_diagnostics(
                        log_dir,
                        rank=rank,
                        payload={
                            "kind": "non_finite_validation_loss",
                            "epoch": epoch,
                            "loss": loss.detach().float().cpu().item(),
                        },
                    )
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch={epoch}"
                    )
                valid_total += loss.detach() * target.shape[0]
                valid_count += target.shape[0]
        row["valid"] = reduce_mean(valid_total, valid_count)

        is_reconstruction_epoch = (epoch + 1) % int(
            config["train"]["validation_panel_interval"]
        ) == 0
        reconstruction_metric = torch.tensor(float("nan"), device=device)
        if rank == 0 and is_reconstruction_epoch:
            panel_model = model.module if hasattr(model, "module") else model
            panel_metrics = validation_panel(
                model=panel_model,
                diffusion=diffusion,
                dataset=valid_dataset,
                config=config,
                device=device,
                seed=validation_seed + 1,
                output_path=panel_dir / f"epoch_{epoch + 1:04d}.npz",
            )
            row.update(panel_metrics)
            metric_key = (
                "panel_reconstruction_l1"
                if reconstruction_selection_metric == "sample_l1"
                else "panel_normalized_mae"
            )
            reconstruction_metric.fill_(float(panel_metrics[metric_key]))
        if world_size > 1 and is_reconstruction_epoch:
            dist.broadcast(reconstruction_metric, src=0)

        row["elapsed_time_per_epoch"] = time.time() - epoch_started
        history.append(row)
        last_epoch = epoch
        if row["valid"] < best_noise_loss:
            best_noise_loss = row["valid"]
            best_noise_epoch = epoch + 1
            if rank == 0:
                atomic_torch_save(model_state(model), best_noise_weights_path)

        if checkpoint_selection == "noise_l1":
            if row["valid"] < best_loss:
                best_loss = row["valid"]
                best_epoch = epoch + 1
                early_stopping_count = 0
                if rank == 0:
                    state = model_state(model)
                    atomic_torch_save(state, best_weights_path)
            else:
                early_stopping_count += 1
        elif is_reconstruction_epoch:
            reconstruction_value = float(reconstruction_metric.item())
            if reconstruction_value < best_reconstruction_loss:
                best_reconstruction_loss = reconstruction_value
                best_reconstruction_epoch = epoch + 1
                best_loss = reconstruction_value
                best_epoch = epoch + 1
                early_stopping_count = 0
                if rank == 0:
                    state = model_state(model)
                    atomic_torch_save(
                        state, best_reconstruction_weights_path
                    )
                    atomic_torch_save(state, best_weights_path)
            else:
                early_stopping_count += 1

        checkpoint_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_noise_epoch": best_noise_epoch,
            "best_noise_loss": best_noise_loss,
            "best_reconstruction_epoch": best_reconstruction_epoch,
            "best_reconstruction_loss": best_reconstruction_loss,
            "checkpoint_selection": checkpoint_selection,
            "reconstruction_selection_metric": reconstruction_selection_metric,
            "early_stopping_count": early_stopping_count,
            "amp_overflow_skips": amp_overflow_skips,
            "consecutive_amp_overflow_skips": consecutive_amp_overflow_skips,
            "model_state_dict": model_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "resolved_config": copy.deepcopy(config),
            "parameter_count": parameter_count,
            "model_signature": (
                f"{config['model']['model_name']}"
                "(x_noisy, condition_lr, obs, noise_level)->epsilon"
            ),
            "diffusion_metadata": copy.deepcopy(config["diffusion"]),
            "seed": seed,
            "last_epoch_seed": epoch_seed,
            "history": copy.deepcopy(history),
        }
        if rank == 0:
            pd.DataFrame(history).to_csv(history_path, index=False)
            print(
                f"epoch={epoch + 1}/{num_epochs} train={row['train']:.8g} "
                f"valid={row['valid']:.8g} elapsed={row['elapsed_time_per_epoch']:.2f}s "
                f"best_epoch={best_epoch} es={early_stopping_count}"
            )
            if (epoch + 1) % args.checkpoint_interval == 0:
                atomic_torch_save(checkpoint_payload, checkpoint_path)
        if world_size > 1:
            dist.barrier()
        if early_stopping_count >= patience:
            break

    if rank == 0 and last_epoch >= start_epoch:
        checkpoint_payload["epoch"] = last_epoch
        atomic_torch_save(checkpoint_payload, checkpoint_path)
        print(f"end={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29831")
    if args.world_size > 1:
        mp.spawn(train_worker_entry, args=(args.world_size, args), nprocs=args.world_size)
    else:
        train_worker_entry(0, 1, args)


def train_worker_entry(
    rank: int, world_size: int, args: argparse.Namespace
) -> None:
    try:
        train_worker(rank, world_size, args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
