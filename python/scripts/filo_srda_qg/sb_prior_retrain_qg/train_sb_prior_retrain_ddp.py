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
from src.filo_srda_qg.sr_prior.data import make_sb_dataset, make_sb_loader
from src.filo_srda_qg.sr_prior.model import make_sb_prior_model
from src.filo_srda_qg.sr_prior.source import build_bridge_source, fold_hr_target
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    make_time_assignment,
)
from src.filo_srda_qg.sb_prior_retrain_qg.ema import EMA
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config
from src.filo_srda_qg.sb_prior_retrain_qg.gates import evaluate_clean_gate
from src.filo_srda_qg.sb_prior_retrain_qg.validation import (
    audit_time_conditioning,
    build_validation_manifest,
    clone_model,
    evaluate_panel,
)
from src.filo_srda_qg.sb_prior_retrain_qg.conditional_runtime import (
    get_conditional_runtime,
)
from src.filo_srda_qg.utils.checkpointing import atomic_torch_save, load_model_state, model_state
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def conditional_mode(config: dict[str, Any]) -> bool:
    return bool(config.get("conditioning", {}).get("enabled", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--override-num-epochs", type=int)
    parser.add_argument("--override-gradient-accumulation-steps", type=int)
    parser.add_argument("--run-name")
    return parser.parse_args()


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    return load_config(path)


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
    return float((total / count.clamp_min(1)).item())


def underlying(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def weight_artifact(
    model: torch.nn.Module,
    *,
    epoch: int,
    weights_kind: str,
    metrics: dict[str, Any],
    artifact_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": {
            "selected_epoch": int(epoch),
            "weights_kind": weights_kind,
            "metrics": metrics,
            **(artifact_metadata or {}),
        },
    }


def flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in values.items()
        if isinstance(value, (int, float, bool, str))
    }


def fail(log_dir: pathlib.Path, rank: int, **payload: Any) -> None:
    (log_dir / f"failure_rank{rank}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)
    if args.run_name:
        config["output"]["run_name"] = args.run_name
    if args.override_num_epochs is not None:
        config["train"]["num_epochs"] = int(args.override_num_epochs)
    if args.override_gradient_accumulation_steps is not None:
        config["train"]["gradient_accumulation_steps"] = int(
            args.override_gradient_accumulation_steps
        )
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

    work_root = root / config["output"]["work_root"]
    run_name = str(config["output"]["run_name"])
    checkpoint_dir = work_root / "checkpoints" / run_name
    log_dir = work_root / "logs" / "training" / run_name
    panel_root = work_root / "samples" / run_name / "validation_panel"
    audit_root = work_root / "audits" / run_name
    for path in (checkpoint_dir, log_dir, panel_root, audit_root):
        path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    history_path = log_dir / "learning_history.csv"
    snapshot_path = checkpoint_dir / "config_snapshot.yml"
    checkpoint_data = (
        torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_path.exists()
        else None
    )
    if checkpoint_data is not None:
        resolved = copy.deepcopy(checkpoint_data["resolved_config"])
        resolved["output"]["run_name"] = run_name
        if args.override_num_epochs is not None:
            resolved["train"]["num_epochs"] = int(args.override_num_epochs)
        if args.override_gradient_accumulation_steps is not None:
            resolved["train"]["gradient_accumulation_steps"] = int(
                args.override_gradient_accumulation_steps
            )
        config = resolved
    if (config.get("training_objective") or {}).get("kind", "drift_mse") != "drift_mse":
        raise ValueError("Expected drift-MSE training")
    if (config.get("initialization") or {}).get("kind", "scratch") != "scratch":
        raise ValueError("Expected scratch initialization or a matching resumable checkpoint")
    conditional_runtime = get_conditional_runtime(config)

    microbatch = int(config["train"]["microbatch_per_gpu"])
    accumulation = int(config["train"]["gradient_accumulation_steps"])
    if microbatch != 1:
        raise ValueError("071B time-sampling contract requires microbatch_per_gpu=1")
    time_assignment = make_time_assignment(config=config, world_size=world_size)
    samples_per_epoch = int(config["train"]["samples_per_epoch"])
    if samples_per_epoch % (world_size * accumulation):
        raise ValueError("samples_per_epoch must divide complete global optimizer groups")

    if conditional_runtime is not None:
        train_dataset = conditional_runtime.make_dataset(
            base_config=config["dataset"], root_dir=root, split="train",
            use_mixup=True, samples_per_epoch=samples_per_epoch,
            deterministic_observations=False,
        )
        valid_dataset = conditional_runtime.make_dataset(
            base_config=config["dataset"], root_dir=root, split="valid",
            use_mixup=False,
            samples_per_epoch=int(config["train"]["valid_samples_per_epoch"]),
            deterministic_observations=True,
        )
        panel_dataset = conditional_runtime.make_dataset(
            base_config=config["dataset"], root_dir=root, split="valid",
            use_mixup=False, samples_per_epoch=None,
            deterministic_observations=True,
        )
    else:
        train_dataset = make_sb_dataset(
            base_config=config["dataset"], root_dir=root, split="train", use_mixup=True,
            samples_per_epoch=samples_per_epoch,
        )
        valid_dataset = make_sb_dataset(
            base_config=config["dataset"], root_dir=root, split="valid", use_mixup=False,
            samples_per_epoch=int(config["train"]["valid_samples_per_epoch"]),
        )
        panel_dataset = make_sb_dataset(
            base_config=config["dataset"], root_dir=root, split="valid", use_mixup=False,
            samples_per_epoch=None,
        )
    panel_dataset.active_indices = list(range(len(panel_dataset.all_hr_paths)))
    train_loader, train_sampler = make_sb_loader(
        dataset=train_dataset, split="train", batch_size=1,
        num_workers=int(config["dataloader"]["num_workers"]), seed=seed,
        rank=rank, world_size=world_size,
    )
    valid_loader, _ = make_sb_loader(
        dataset=valid_dataset, split="valid", batch_size=1, num_workers=0,
        seed=seed + 10_000, rank=rank, world_size=world_size,
    )
    if len(train_loader) % accumulation:
        raise ValueError("Per-rank training loader must contain complete accumulation groups")

    if conditional_runtime is not None:
        model = conditional_runtime.make_model(config["model"]).to(device)
    else:
        model = make_sb_prior_model(config["model"]).to(device)
    initialization_metadata: dict[str, Any] = {"kind": "scratch"}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    artifact_metadata = {
        "parameter_count": parameter_count,
        "model_signature": (
            conditional_runtime.model_signature
            if conditional_runtime is not None
            else "SBPriorUNet3D:18->9"
        ),
        "observation_contract": (
            {
                "enabled": True,
                **conditional_runtime.observation_contract,
                "future_slot_missing": bool(
                    config["dataset"]["is_future_obs_missing"]
                ),
                "grid_intervals": [
                    int(config["dataset"]["obs_grid_interval_x"]),
                    int(config["dataset"]["obs_grid_interval_y"]),
                ],
                "physical_noise_std": float(config["dataset"]["obs_noise_std"]),
            }
            if conditional_runtime is not None
            else {"enabled": False}
        ),
        "initialization": initialization_metadata,
    }
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if cuda else None)
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(**config["bridge"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    ema = EMA(underlying(model), decay=float(config["train"]["ema_decay"]))
    precision = str(config["train"]["precision"])
    amp_enabled = cuda and precision == "bfloat16"
    precision_metadata = {
        "requested": precision,
        "actual": precision,
        "bf16_autocast": bool(amp_enabled),
        "fallback_reason": None,
    }
    if amp_enabled and not torch.cuda.is_bf16_supported():
        amp_enabled = False
        precision_metadata = {
            "requested": precision,
            "actual": "float32",
            "bf16_autocast": False,
            "fallback_reason": "torch.cuda.is_bf16_supported() returned False",
        }

    start_epoch = 0
    best_epoch = 0
    best_panel_crps_raw_epoch = 0
    best_panel_crps_ema_epoch = 0
    best_clean_raw_epoch = 0
    best_clean_ema_epoch = 0
    best_loss = float("inf")
    best_panel_crps_raw = float("inf")
    best_panel_crps_ema = float("inf")
    best_clean_crps_raw = float("inf")
    best_clean_crps_ema = float("inf")
    early_stopping_count = 0
    history: list[dict[str, Any]] = []
    panel_history: list[dict[str, Any]] = []
    audit_history: list[dict[str, Any]] = []
    if checkpoint_data is not None:
        initialization_metadata = dict(
            checkpoint_data.get("initialization_metadata", initialization_metadata)
        )
        artifact_metadata["initialization"] = initialization_metadata
        load_model_state(model, checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        ema.load_state_dict(checkpoint_data["ema_state_dict"])
        start_epoch = int(checkpoint_data["epoch"]) + 1
        best_epoch = int(checkpoint_data.get("best_epoch", 0))
        best_loss = float(checkpoint_data.get("best_loss", float("inf")))
        legacy_panel_epoch = int(checkpoint_data.get("best_panel_crps_epoch", 0))
        legacy_panel_crps = float(checkpoint_data.get("best_panel_crps", float("inf")))
        legacy_clean_epoch = int(checkpoint_data.get("best_clean_epoch", 0))
        legacy_clean_crps = float(checkpoint_data.get("best_clean_crps", float("inf")))
        best_panel_crps_raw_epoch = int(
            checkpoint_data.get("best_panel_crps_raw_epoch", legacy_panel_epoch)
        )
        best_panel_crps_ema_epoch = int(
            checkpoint_data.get("best_panel_crps_ema_epoch", legacy_panel_epoch)
        )
        best_panel_crps_raw = float(
            checkpoint_data.get("best_panel_crps_raw", legacy_panel_crps)
        )
        best_panel_crps_ema = float(
            checkpoint_data.get("best_panel_crps_ema", legacy_panel_crps)
        )
        best_clean_raw_epoch = int(
            checkpoint_data.get("best_clean_raw_epoch", legacy_clean_epoch)
        )
        best_clean_ema_epoch = int(
            checkpoint_data.get("best_clean_ema_epoch", legacy_clean_epoch)
        )
        best_clean_crps_raw = float(
            checkpoint_data.get("best_clean_crps_raw", legacy_clean_crps)
        )
        best_clean_crps_ema = float(
            checkpoint_data.get("best_clean_crps_ema", legacy_clean_crps)
        )
        early_stopping_count = int(checkpoint_data.get("early_stopping_count", 0))
        history = list(checkpoint_data.get("history", []))
        panel_history = list(checkpoint_data.get("panel_history", []))
        audit_history = list(checkpoint_data.get("audit_history", []))

    if rank == 0:
        with snapshot_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        manifest = build_validation_manifest(
            panel_dataset,
            config["validation_panel"]["starts"],
            audit_root / "validation_manifest.json",
        )
    else:
        manifest = []
    if dist.is_initialized():
        payload = [manifest]
        dist.broadcast_object_list(payload, src=0)
        manifest = payload[0]

    if rank == 0:
        print(f"start={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        print(f"host={socket.gethostname()} run_name={run_name}")
        print(f"parameter_count={parameter_count:,} world_size={world_size}")
        print(
            f"microbatch=1 accumulation={accumulation} "
            f"effective_global_batch={world_size * accumulation} "
            f"precision={precision_metadata['actual']} "
            f"requested_precision={precision_metadata['requested']} "
            f"noise_exponent={bridge.config.noise_exponent} epsilon={bridge.config.epsilon} "
            f"time_mode={time_assignment.mode}"
        )
        print(f"initialization={json.dumps(initialization_metadata, default=str)}")
        if precision_metadata["fallback_reason"]:
            print(f"precision_fallback_reason={precision_metadata['fallback_reason']}")

    if dist.is_initialized():
        dist.barrier()

    num_epochs = int(config["train"]["num_epochs"])
    for epoch in range(start_epoch, num_epochs):
        started = time.perf_counter()
        reset_rng(seed + 1_000_000 * epoch + rank)
        train_dataset.set_epoch(epoch)
        valid_dataset.set_epoch(0)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = torch.tensor(0.0, device=device)
        train_count = torch.tensor(0.0, device=device)
        last_grad_norm = 0.0
        for batch_index, batch in enumerate(train_loader):
            group = batch_index // accumulation
            accumulation_index = batch_index % accumulation
            t, role = time_assignment.time(
                epoch=epoch, optimizer_group=group, rank=rank,
                accumulation_index=accumulation_index, device=device,
            )
            source = build_bridge_source(batch["condition_lr"].to(device, non_blocking=True))
            target = fold_hr_target(batch["hr"].to(device, non_blocking=True))
            observations = batch["obs"].to(device, non_blocking=True) if conditional_runtime is not None else None
            masks = batch["mask"].to(device, non_blocking=True) if conditional_runtime is not None else None
            if role.kind == "interior":
                noise = torch.randn_like(source)
            else:
                noise = torch.zeros_like(source)
            should_step = accumulation_index + 1 == accumulation
            drift_sync = (
                model.no_sync()
                if hasattr(model, "no_sync") and not should_step
                else contextlib.nullcontext()
            )
            with drift_sync:
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    if conditional_runtime is not None:
                        raw_loss = conditional_runtime.loss_at_time(
                            bridge, model, source=source, target=target,
                            observations=observations,
                            masks=masks,
                            t=t, noise=noise,
                        )
                    else:
                        raw_loss = bridge.loss_at_time(
                            model, source=source, target=target, t=t, noise=noise
                        )
                    drift_loss = raw_loss / accumulation
                if not bool(torch.isfinite(raw_loss) & torch.isfinite(drift_loss)):
                    fail(log_dir, rank, kind="non_finite_loss", epoch=epoch, batch=batch_index)
                    raise FloatingPointError("non-finite training loss")
                drift_loss.backward()
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["train"]["gradient_clip_norm"])
                )
                finite = torch.tensor(int(torch.isfinite(grad_norm)), device=device)
                if dist.is_initialized():
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    fail(log_dir, rank, kind="non_finite_gradient", epoch=epoch, batch=batch_index)
                    raise FloatingPointError("non-finite gradient")
                optimizer.step()
                ema.update(underlying(model))
                optimizer.zero_grad(set_to_none=True)
                last_grad_norm = float(grad_norm.item())
            train_total += raw_loss.detach()
            train_count += 1

        underlying(model).eval()
        if conditional_mode(config):
            # Hold validation masks/noise fixed across epochs and resumes.
            reset_rng(seed + 8_000_000 + rank)
        valid_total = torch.tensor(0.0, device=device)
        valid_count = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for local_index, batch in enumerate(valid_loader):
                global_index = local_index * world_size + rank
                t, role = time_assignment.deterministic_validation_time(
                    global_index, device=device, dtype=torch.float32
                )
                source = build_bridge_source(batch["condition_lr"].to(device))
                target = fold_hr_target(batch["hr"].to(device))
                if role.kind == "interior":
                    generator = torch.Generator(device=device).manual_seed(
                        seed + 700_001 + global_index
                    )
                    noise = torch.randn(source.shape, device=device, generator=generator)
                else:
                    noise = torch.zeros_like(source)
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    if conditional_runtime is not None:
                        value = conditional_runtime.loss_at_time(
                            bridge, underlying(model), source=source, target=target,
                            observations=batch["obs"].to(device),
                            masks=batch["mask"].to(device), t=t, noise=noise,
                        )
                    else:
                        value = bridge.loss_at_time(
                            underlying(model), source=source, target=target, t=t, noise=noise
                        )
                valid_total += value
                valid_count += 1
        valid_loss = reduce_mean(valid_total, valid_count)
        train_loss = reduce_mean(train_total, train_count)
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": train_loss,
            "valid": valid_loss,
            "gradient_norm": last_grad_norm,
        }

        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch + 1
            if epoch + 1 >= int(config["train"]["early_stopping_min_epoch"]):
                early_stopping_count = 0
            if rank == 0:
                atomic_torch_save(
                    weight_artifact(
                        underlying(model), epoch=epoch + 1, weights_kind="raw",
                        metrics={"validation_loss": valid_loss},
                        artifact_metadata=artifact_metadata,
                    ),
                    checkpoint_dir / "best_loss_raw.pt",
                )
        elif epoch + 1 >= int(config["train"]["early_stopping_min_epoch"]):
            early_stopping_count += 1

        panel_due = (epoch + 1) % int(config["validation_panel"]["interval"]) == 0
        convergence_due = (epoch + 1) % int(config["validation_panel"]["convergence_interval"]) == 0
        if rank == 0 and panel_due:
            raw_eval = underlying(model).float().eval()
            ema_eval = clone_model(raw_eval).float().eval()
            ema.copy_to(ema_eval)
            epoch_dir = panel_root / f"epoch_{epoch + 1:04d}"
            common = dict(
                dataset=panel_dataset, manifest=manifest, bridge=bridge,
                ensemble_size=int(config["validation_panel"]["ensemble_size"]),
                member_chunk_size=int(config["validation_panel"]["member_chunk_size"]),
                brownian_base_steps=int(config["validation_panel"]["brownian_base_steps"]),
                sampling_seed=int(config["validation_panel"]["sampling_seed"]),
                pv_min=float(config["dataset"]["pv_min"]), pv_max=float(config["dataset"]["pv_max"]),
            )
            if conditional_runtime is not None:
                conditional_common = {
                    **common,
                    "observation_seed": int(
                        config["validation_panel"].get("observation_seed", 7242)
                    ),
                }
                raw20 = conditional_runtime.evaluate_panel(
                    model=raw_eval, nfe_budget=20,
                    output_dir=epoch_dir / "raw_nfe20", weights_kind="raw",
                    **conditional_common,
                )
                ema20 = conditional_runtime.evaluate_panel(
                    model=ema_eval, nfe_budget=20,
                    output_dir=epoch_dir / "ema_nfe20", weights_kind="ema",
                    **conditional_common,
                )
            else:
                raw20 = evaluate_panel(
                    model=raw_eval, nfe_budget=20,
                    output_dir=epoch_dir / "raw_nfe20", weights_kind="raw", **common
                )
                ema20 = evaluate_panel(
                    model=ema_eval, nfe_budget=20,
                    output_dir=epoch_dir / "ema_nfe20", weights_kind="ema", **common
                )
            panel_record: dict[str, Any] = {
                "epoch": epoch + 1,
                **flatten("raw20", raw20),
                **flatten("ema20", ema20),
            }
            panel_history.append(panel_record)
            row.update(flatten("panel_raw20", raw20))
            row.update(flatten("panel_ema20", ema20))
            if float(raw20["physical_crps"]) < best_panel_crps_raw:
                best_panel_crps_raw = float(raw20["physical_crps"])
                best_panel_crps_raw_epoch = epoch + 1
                atomic_torch_save(
                    weight_artifact(
                        raw_eval,
                        epoch=epoch + 1,
                        weights_kind="raw",
                        metrics=raw20,
                        artifact_metadata=artifact_metadata,
                    ),
                    checkpoint_dir / "best_panel_crps_raw.pt",
                )
            if float(ema20["physical_crps"]) < best_panel_crps_ema:
                best_panel_crps_ema = float(ema20["physical_crps"])
                best_panel_crps_ema_epoch = epoch + 1
                atomic_torch_save(
                    weight_artifact(
                        ema_eval,
                        epoch=epoch + 1,
                        weights_kind="ema",
                        metrics=ema20,
                        artifact_metadata=artifact_metadata,
                    ),
                    checkpoint_dir / "best_panel_crps_ema.pt",
                )
            if convergence_due:
                if conditional_runtime is not None:
                    raw50 = conditional_runtime.evaluate_panel(
                        model=raw_eval, nfe_budget=50,
                        output_dir=epoch_dir / "raw_nfe50", weights_kind="raw",
                        **conditional_common,
                    )
                    ema50 = conditional_runtime.evaluate_panel(
                        model=ema_eval, nfe_budget=50,
                        output_dir=epoch_dir / "ema_nfe50", weights_kind="ema",
                        **conditional_common,
                    )
                    audit_args = dict(
                        dataset=panel_dataset, manifest=manifest, bridge=bridge,
                        noise_repeats=int(config["time_audit"]["noise_repeats"]),
                        seed=int(config["time_audit"]["seed"]),
                        observation_seed=int(
                            config["validation_panel"].get("observation_seed", 7242)
                        ),
                    )
                    raw_audit = conditional_runtime.audit_time(
                        model=raw_eval, **audit_args
                    )
                    ema_audit = conditional_runtime.audit_time(
                        model=ema_eval, **audit_args
                    )
                else:
                    raw50 = evaluate_panel(
                        model=raw_eval, nfe_budget=50,
                        output_dir=epoch_dir / "raw_nfe50", weights_kind="raw", **common
                    )
                    ema50 = evaluate_panel(
                        model=ema_eval, nfe_budget=50,
                        output_dir=epoch_dir / "ema_nfe50", weights_kind="ema", **common
                    )
                    raw_audit = audit_time_conditioning(
                        model=raw_eval, dataset=panel_dataset, manifest=manifest,
                        bridge=bridge,
                        noise_repeats=int(config["time_audit"]["noise_repeats"]),
                        seed=int(config["time_audit"]["seed"]),
                    )
                    ema_audit = audit_time_conditioning(
                        model=ema_eval, dataset=panel_dataset, manifest=manifest,
                        bridge=bridge,
                        noise_repeats=int(config["time_audit"]["noise_repeats"]),
                        seed=int(config["time_audit"]["seed"]),
                    )
                raw_gate = evaluate_clean_gate(
                    nfe20=raw20, nfe50=raw50,
                    midpoint_grid_ratio=float(raw_audit["midpoint_grid_ratio"]),
                    thresholds=config["clean_gate"],
                )
                ema_gate = evaluate_clean_gate(
                    nfe20=ema20, nfe50=ema50,
                    midpoint_grid_ratio=float(ema_audit["midpoint_grid_ratio"]),
                    thresholds=config["clean_gate"],
                )
                audit_entry = {
                    "epoch": epoch + 1,
                    "raw": {key: value for key, value in raw_audit.items() if key != "rows"},
                    "ema": {key: value for key, value in ema_audit.items() if key != "rows"},
                    "raw_gate_passed": raw_gate.passed,
                    "raw_gate_checks": raw_gate.checks,
                    "raw_gate_values": raw_gate.values,
                    "ema_gate_passed": ema_gate.passed,
                    "ema_gate_checks": ema_gate.checks,
                    "ema_gate_values": ema_gate.values,
                }
                audit_history.append(audit_entry)
                (epoch_dir / "time_and_clean_gate.json").write_text(
                    json.dumps(audit_entry, indent=2)
                )
                pd.DataFrame(ema_audit["rows"]).to_csv(
                    epoch_dir / "ema_time_audit_samples.csv", index=False
                )
                panel_record.update(flatten("raw50", raw50))
                panel_record.update(flatten("ema50", ema50))
                panel_record.update(
                    raw_gate_passed=raw_gate.passed,
                    ema_gate_passed=ema_gate.passed,
                    **flatten("raw_gate", raw_gate.values),
                    **flatten("ema_gate", ema_gate.values),
                )
                row.update(flatten("panel_raw50", raw50))
                row.update(flatten("panel_ema50", ema50))
                row.update(
                    raw_gate_passed=raw_gate.passed,
                    ema_gate_passed=ema_gate.passed,
                    **flatten("raw_gate", raw_gate.values),
                    **flatten("ema_gate", ema_gate.values),
                )
                if raw_gate.passed and float(raw50["physical_crps"]) < best_clean_crps_raw:
                    best_clean_crps_raw = float(raw50["physical_crps"])
                    best_clean_raw_epoch = epoch + 1
                    atomic_torch_save(
                        weight_artifact(
                            raw_eval,
                            epoch=epoch + 1,
                            weights_kind="raw",
                            metrics=raw50,
                            artifact_metadata=artifact_metadata,
                        ),
                        checkpoint_dir / "best_panel_clean_raw.pt",
                    )
                if ema_gate.passed and float(ema50["physical_crps"]) < best_clean_crps_ema:
                    best_clean_crps_ema = float(ema50["physical_crps"])
                    best_clean_ema_epoch = epoch + 1
                    atomic_torch_save(
                        weight_artifact(
                            ema_eval,
                            epoch=epoch + 1,
                            weights_kind="ema",
                            metrics=ema50,
                            artifact_metadata=artifact_metadata,
                        ),
                        checkpoint_dir / "best_panel_clean_ema.pt",
                    )
            del ema_eval

        row["elapsed_time_per_epoch"] = time.perf_counter() - started
        row["early_stopping_count"] = early_stopping_count
        row["best_loss_epoch"] = best_epoch
        row["best_panel_crps_raw_epoch"] = best_panel_crps_raw_epoch
        row["best_panel_crps_ema_epoch"] = best_panel_crps_ema_epoch
        row["best_clean_raw_epoch"] = best_clean_raw_epoch
        row["best_clean_ema_epoch"] = best_clean_ema_epoch
        history.append(row)
        checkpoint_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_panel_crps_raw_epoch": best_panel_crps_raw_epoch,
            "best_panel_crps_ema_epoch": best_panel_crps_ema_epoch,
            "best_panel_crps_raw": best_panel_crps_raw,
            "best_panel_crps_ema": best_panel_crps_ema,
            "best_clean_raw_epoch": best_clean_raw_epoch,
            "best_clean_ema_epoch": best_clean_ema_epoch,
            "best_clean_crps_raw": best_clean_crps_raw,
            "best_clean_crps_ema": best_clean_crps_ema,
            "best_panel_crps_epoch": best_panel_crps_ema_epoch,
            "best_panel_crps": best_panel_crps_ema,
            "best_clean_epoch": best_clean_ema_epoch,
            "best_clean_crps": best_clean_crps_ema,
            "early_stopping_count": early_stopping_count,
            "model_state_dict": model_state(model),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "resolved_config": config,
            "parameter_count": parameter_count,
            "model_signature": (
                conditional_runtime.model_signature
                if conditional_runtime is not None
                else "SBPriorUNet3D:18->9"
            ),
            "precision_metadata": precision_metadata,
            "initialization_metadata": initialization_metadata,
            "time_sampling_contract": {
                "mode": time_assignment.mode,
                "global_batch_size": time_assignment.global_batch_size,
                "interior_strata": time_assignment.interior_strata,
                "dense_grid_intervals": time_assignment.dense_grid_intervals,
                "endpoint_roles": [0.0, 1.0],
                "t1_target": "conditional_mean",
            },
            "observation_contract": (
                {
                    "enabled": True,
                    **conditional_runtime.observation_contract,
                    "future_slot_missing": bool(
                        config["dataset"]["is_future_obs_missing"]
                    ),
                    "grid_intervals": [
                        int(config["dataset"]["obs_grid_interval_x"]),
                        int(config["dataset"]["obs_grid_interval_y"]),
                    ],
                    "physical_noise_std": float(config["dataset"]["obs_noise_std"]),
                }
                if conditional_runtime is not None
                else {"enabled": False}
            ),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if cuda else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "history": history,
            "panel_history": panel_history,
            "audit_history": audit_history,
        }
        if rank == 0:
            atomic_torch_save(checkpoint_payload, checkpoint_path)
            if (epoch + 1) % int(config["train"]["checkpoint_interval"]) == 0:
                atomic_torch_save(
                    checkpoint_payload,
                    checkpoint_dir / f"checkpoint_epoch{epoch + 1:04d}.pth",
                )
            pd.DataFrame(history).to_csv(history_path, index=False)
            pd.DataFrame(panel_history).to_csv(log_dir / "panel_history.csv", index=False)
            (log_dir / "audit_history.json").write_text(json.dumps(audit_history, indent=2))
            print(
                f"epoch={epoch + 1}/{num_epochs} train={train_loss:.8g} valid={valid_loss:.8g} "
                f"grad={last_grad_norm:.4g} elapsed={row['elapsed_time_per_epoch']:.2f}s "
                f"best_loss={best_epoch} "
                f"best_panel_raw={best_panel_crps_raw_epoch} best_panel_ema={best_panel_crps_ema_epoch} "
                f"best_clean_raw={best_clean_raw_epoch} best_clean_ema={best_clean_ema_epoch} "
                f"es={early_stopping_count}"
            )
        if dist.is_initialized():
            dist.barrier()
        if (
            epoch + 1 >= int(config["train"]["early_stopping_min_epoch"])
            and early_stopping_count >= int(config["train"]["early_stopping_patience"])
        ):
            if rank == 0:
                print("early stopping")
            break
    if rank == 0:
        print(f"end={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29781")
    if args.world_size > 1:
        mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size)
    else:
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
