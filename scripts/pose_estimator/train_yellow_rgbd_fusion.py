"""Train the first single-frame RGB-D fusion on completed teacher-labelled pairs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from datetime import timedelta
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler, Subset
from viserdex.collectors.rgbd_fusion_network import RGBDKeypointNetwork
from viserdex.utils.pose_metrics import fit_rigid_pose


from viserdex.collectors.yellow_fusion_data import YellowDataset, verify_dataset


def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    values = {"keypoint_mm": [], "translation_mm": [], "rotation_deg": []}
    for batch in loader:
        b = batch_to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(b["rgb"], b["points"], b["uv"], b["valid"])
        pr, pt = fit_rigid_pose(model.reference.double(), pred.double())
        gr, gt = fit_rigid_pose(model.reference.double(), b["target"].double())
        angle = torch.rad2deg(
            torch.acos(((pr.transpose(-1, -2) @ gr).diagonal(dim1=-2, dim2=-1).sum(-1) - 1).div(2).clamp(-1, 1))
        )
        values["keypoint_mm"].extend(((pred - b["target"]).norm(dim=-1).mean(-1) * 1000).cpu().tolist())
        values["translation_mm"].extend(((pt - gt).norm(dim=-1) * 1000).cpu().tolist())
        values["rotation_deg"].extend(angle.cpu().tolist())
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, values)
        values = {key: [value for group in gathered for value in group[key]] for key in values}
    result = {}
    for key, rows in values.items():
        v = torch.tensor(rows, dtype=torch.float64)
        result[key] = float(v.mean())
        result[key + "_p95"] = float(v.quantile(0.95))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--max_train_batches", type=int, default=0, help="Smoke test only; 0 uses the full dataset.")
    parser.add_argument("--rgb_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--init_fusion_checkpoint", type=Path)
    parser.add_argument(
        "--resume_checkpoint", type=Path, help="Restore model, optimizer, scheduler, epoch and update count."
    )
    parser.add_argument("--freeze_epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Per-rank batch; global batch = batch_size * world size."
    )
    parser.add_argument("--workers", type=int, default=4, help="Loader workers per rank.")
    parser.add_argument(
        "--batch_norm", choices=["sync", "local"], default="sync", help="BatchNorm statistics for distributed training."
    )
    parser.add_argument("--amp", choices=["none", "bf16"], default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile_steps", type=int, default=0, help="Capture CPU/CUDA traces for this many warmed-up updates."
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or min(args.workers, args.freeze_epochs, args.profile_steps) < 0:
        parser.error("Invalid counts")
    if args.resume_checkpoint and args.init_fusion_checkpoint:
        parser.error("Choose model initialization or full training resume")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world > 1
    if distributed:
        if args.freeze_epochs:
            parser.error("Use freeze_epochs=0 for DDP after the single-GPU warm-up")
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    torch.set_num_threads(2 if distributed else 4)
    torch.cuda.set_device(args.device)
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=torch.device(args.device))
    amp = args.amp == "bf16"
    if amp and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 not supported by selected device")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    verification = verify_dataset(args.dataset_root)
    train = YellowDataset(args.dataset_root, "train")
    val = YellowDataset(args.dataset_root, "validation")
    if not np.array_equal(train.reference, val.reference):
        raise ValueError("Training/validation reference mismatch")
    first = train
    out = args.output_dir.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()
    loader_args = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )
    sampler = DistributedSampler(train, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, **loader_args)
    # Validation has no padded/repeated examples, and bypasses DDP forward collectives.
    val_subset = Subset(val, range(rank, len(val), world)) if distributed else val
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    raw_model = RGBDKeypointNetwork(first.reference, args.rgb_checkpoint).to(args.device)
    previous = None
    checkpoint_path = args.resume_checkpoint or args.init_fusion_checkpoint
    if checkpoint_path:
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not torch.equal(previous["reference"], raw_model.reference.cpu()):
            raise ValueError("Fusion initialization uses different canonical keypoints")
        raw_model.load_state_dict(previous["model"], strict=True)
    if distributed:
        if args.batch_norm == "sync":
            raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
        model = DistributedDataParallel(raw_model, device_ids=[torch.cuda.current_device()])
    else:
        model = raw_model
    backbone = list(raw_model.rgb_features.parameters())
    backbone_ids = {id(p) for p in backbone}
    added = [p for p in raw_model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": 1e-5}, {"params": added, "lr": 1e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch = 0
    schedule_extension = None
    if args.resume_checkpoint:
        if previous["launch"]["dataset_verification"] != verification:
            raise ValueError("Resume requires identical dataset manifests, indices and validation reports")
        if previous["scheduler"]["T_max"] > args.epochs:
            raise ValueError("Resume cannot shorten the saved cosine schedule")
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        start_epoch = previous["epoch"]
        if args.epochs > scheduler.T_max:
            schedule_extension = dict(
                from_epochs=scheduler.T_max,
                to_epochs=args.epochs,
                at_epoch=start_epoch,
                learning_rates=[g["lr"] for g in optimizer.param_groups],
                policy="extend cosine smoothly without changing the restored current learning rate",
            )
            scheduler.T_max = args.epochs
            factor = (1 + math.cos(math.pi * start_epoch / args.epochs)) / 2
            scheduler.base_lrs = [
                scheduler.eta_min + (g["lr"] - scheduler.eta_min) / factor for g in optimizer.param_groups
            ]
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["initial_lr"] = base_lr
        if not 0 <= start_epoch < args.epochs:
            raise ValueError("Resume epoch is outside the requested training range")
    launch = dict(
        arguments={
            k: [str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        train_samples=len(train),
        val_samples=len(val),
        dataset_verification=verification,
        input_keys={"train_rgb": "rgb", "validation_rgb": "rgb_clean", "points": "points_camera_clean_m"},
        uv_convention="continuous image coordinates, no additional half-pixel offset",
        reference_keypoints_m=first.reference.tolist(),
        physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        checkpoint_selection="lowest clean validation mean keypoint error",
        smoke_test=bool(args.max_train_batches),
        optimizer_config={"backbone_lr": 1e-5, "fusion_lr": 1e-4, "weight_decay": 1e-4,
                          "cosine_eta_min": 1e-6, "smooth_l1_beta": 0.1, "gradient_clip": 5.0},
        rgb_checkpoint_sha256=hashlib.sha256(args.rgb_checkpoint.read_bytes()).hexdigest(),
        network="ResNet34 spatial features + corresponding RGB/PointNet fusion + 9 semantic XYZ keypoints",
        normalization="observed visible-point centroid only; no truth centering",
        rgb_size=[640, 360],
        depth_size=[848, 480],
        points=1024,
        augmentations=["capture-time ViserDex SH augmentation", "brightness 0.75-1.25/contrast 0.85-1.15", "fresh 0.5mm ray-depth noise clipped +/-1.5mm from CLEAN points", "15% point dropout"],
        limitation="simulation only; virtual depth camera; real-device accuracy not validated",
        world_size=world,
        global_batch_size=args.batch_size * world,
        sync_batch_norm=distributed and args.batch_norm == "sync",
        precision=args.amp,
        device=torch.cuda.get_device_name(args.device),
        torch_version=str(torch.__version__),
        backbone_frozen_epochs=args.freeze_epochs,
        fusion_init_sha256=hashlib.sha256(args.init_fusion_checkpoint.read_bytes()).hexdigest()
        if args.init_fusion_checkpoint
        else None,
        optimizer_initialization="restored AdamW and cosine schedule"
        if args.resume_checkpoint
        else "fresh AdamW/schedule; fixed base learning rates",
        resumed_epoch=start_epoch,
        schedule_extension=schedule_extension,
        resume_checkpoint_sha256=hashlib.sha256(args.resume_checkpoint.read_bytes()).hexdigest()
        if args.resume_checkpoint
        else None,
        semantic_symmetry_minimization=False,
    )
    if rank == 0:
        (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        if args.resume_checkpoint and (args.resume_checkpoint.parent / "best.pt").exists():
            shutil.copy2(args.resume_checkpoint.parent / "best.pt", out / "best.pt")
    start = time.monotonic()
    step = previous["step"] if args.resume_checkpoint else 0
    best = (
        previous.get("best_keypoint_mm", previous["metrics"]["keypoint_mm"]) if args.resume_checkpoint else float("inf")
    )

    if args.resume_checkpoint and "best_keypoint_mm" not in previous:
        old_log = args.resume_checkpoint.parent / "metrics.jsonl"
        if old_log.exists():
            best = min(
                [best]
                + [
                    r["keypoint_mm"]
                    for r in map(json.loads, old_log.read_text().splitlines())
                    if r["kind"] == "validation" and r["epoch"] <= start_epoch
                ]
            )

    def save(name, epoch, metrics):
        if rank:
            return
        temp = out / (name + ".tmp")
        torch.save(
            dict(
                model=raw_model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                step=step,
                best_keypoint_mm=best,
                metrics=metrics,
                reference=raw_model.reference.cpu(),
                launch=launch,
            ),
            temp,
        )
        os.replace(temp, out / name)

    log = (out / "metrics.jsonl").open("a", buffering=1) if rank == 0 else None
    initial = evaluate(raw_model, val_loader, args.device, amp)
    if rank == 0:
        print("INITIAL_VALIDATION", json.dumps(initial), flush=True)
        log.write(json.dumps(dict(kind="initial_validation", step=0, **initial)) + "\n")
    profiler = None
    if args.profile_steps:

        def export_profile(profile):
            profile.export_chrome_trace(str(out / f"profile_rank{rank}.json"))
            summary = profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
            (out / f"profile_rank{rank}.txt").write_text(summary)

        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=3, warmup=2, active=args.profile_steps, repeat=1),
            on_trace_ready=export_profile,
        )
        profiler.start()

    def phase(name):
        return torch.profiler.record_function(name) if profiler else nullcontext()

    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        frozen = epoch < args.freeze_epochs
        raw_model.rgb_features.requires_grad_(not frozen)
        if frozen:
            raw_model.rgb_features.eval()
        loss_sum = torch.zeros(2, device=args.device, dtype=torch.float64)
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            with phase("phase.host_to_device"):
                b = batch_to_device(batch, args.device)
            with phase("phase.augmentation"):
                brightness = torch.empty(len(b["rgb"]), 1, 1, 1, device=args.device).uniform_(0.75, 1.25)
                contrast = torch.empty_like(brightness).uniform_(0.85, 1.15)
                foreground = b["rgb"].any(dim=1, keepdim=True)
                augmented = ((b["rgb"] - 0.5) * contrast + 0.5).mul(brightness).clamp(0, 1)
                b["rgb"] = torch.where(foreground, augmented, 0.0)
                noise = (torch.randn_like(b["points"][..., 2:]) * 0.0005).clamp(-0.0015, 0.0015)
                b["points"] = b["points"] * (1 + noise / b["points"][..., 2:].clamp_min(0.1))
                original_valid = b["valid"]
                b["valid"] = original_valid & (torch.rand_like(original_valid.float()) > 0.15)
                empty = ~b["valid"].any(-1)
                b["valid"][empty] = original_valid[empty]
            optimizer.zero_grad(set_to_none=True)
            with phase("phase.forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    pred = model(b["rgb"], b["points"], b["uv"], b["valid"])
                    loss = F.smooth_l1_loss(pred / raw_model.edge, b["target"] / raw_model.edge, beta=0.1)
            with phase("phase.finite_loss_sync"):
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
            with phase("phase.backward"):
                loss.backward()
            with phase("phase.optimizer"):
                norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            step += 1
            loss_sum[0] += loss.detach() * len(b["rgb"])
            loss_sum[1] += len(b["rgb"])
            if step % 10 == 0 or step == 1:
                averaged = loss.detach().clone()
                if distributed:
                    dist.all_reduce(averaged)
                    averaged /= world
                if rank == 0:
                    record = dict(
                        kind="train",
                        epoch=epoch + 1,
                        step=step,
                        loss=float(averaged),
                        grad_norm=float(norm),
                        elapsed_s=time.monotonic() - start,
                    )
                    log.write(json.dumps(record) + "\n")
                    print("TRAIN", json.dumps(record), flush=True)
            if profiler:
                profiler.step()
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - epoch_started
        if distributed:
            dist.all_reduce(loss_sum)
            duration = torch.tensor(train_seconds, device=args.device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            train_seconds = float(duration)
        scheduler.step()
        metrics = evaluate(raw_model, val_loader, args.device, amp)
        record = dict(
            kind="validation",
            epoch=epoch + 1,
            step=step,
            train_loss=float(loss_sum[0] / loss_sum[1]),
            train_seconds=train_seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            training_samples_per_s=float(loss_sum[1]) / train_seconds,
            world_size=world,
            global_batch_size=args.batch_size * world,
            precision=args.amp,
            elapsed_s=time.monotonic() - start,
            **metrics,
        )
        if rank == 0:
            log.write(json.dumps(record) + "\n")
            print("VALIDATION", json.dumps(record), flush=True)
        improved = metrics["keypoint_mm"] < best
        best = min(best, metrics["keypoint_mm"])
        save("latest.pt", epoch + 1, metrics)
        if improved:
            save("best.pt", epoch + 1, metrics)
        if rank == 0:
            (out / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
    if profiler:
        profiler.stop()
    save("final.pt", args.epochs, metrics)
    if rank == 0:
        (out / "completed.json").write_text(
            json.dumps(dict(completed=True, smoke_test=bool(args.max_train_batches), epochs=args.epochs, steps=step, best_keypoint_mm=best, final=metrics), indent=2) + "\n"
        )
        log.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
