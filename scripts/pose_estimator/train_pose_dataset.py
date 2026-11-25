"""Train the existing pose CNN on completed, teacher-labelled Revo3 occlusion datasets."""

import argparse
import importlib.util
import json
import math
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from viserdex.utils.pose_loss import keypoint_loss
from viserdex.utils.pose_metrics import pose_metrics
from viserdex.utils.early_stopping import EarlyStopping


def load_module(name, path):
    # Import the reusable CNN without importing tasks/__init__ (which starts the
    # Isaac Lab dependency chain). Offline training needs no simulator or renderer.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OccludedDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if not self.manifest["completed"] or self.manifest["hand"] != "Revo3":
            raise ValueError(f"Expected a completed Revo3 dataset: {directory}")
        with h5py.File(self.directory / "samples.h5", "r") as data:
            self.length = len(data["rgb"])
            assert self.length == self.manifest["samples"] > 0
            assert data["rgb"].shape[1:] == (120, 120, 3)
            fractions = data["occlusion_fraction"][:]
            assert np.isfinite(fractions).all() and (fractions >= 0.02).all()
            assert np.isfinite(data["keypoints_uvd"][:]).all()
        self.handle = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.directory / "samples.h5", "r")
        return (
            torch.from_numpy(self.handle["rgb"][index]).permute(2, 0, 1),
            torch.from_numpy(self.handle["keypoints_uvd"][index]),
            np.float32(self.handle["occlusion_fraction"][index]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_lr_factor", type=float, default=0.2)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init_checkpoint", help="Fine-tune weights in a new run with a fresh optimizer and schedule.")
    parser.add_argument(
        "--early_stop_patience", type=int, default=0, help="Epochs without meaningful ADD improvement; 0 disables."
    )
    parser.add_argument("--min_relative_improvement", type=float, default=0.01)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_interval, args.log_interval) <= 0 or args.workers < 0:
        parser.error("Counts must be positive; workers may be zero")
    if args.resume and args.init_checkpoint:
        parser.error("Use resume or init_checkpoint, not both")
    if args.early_stop_patience < 0 or not 0 <= args.min_relative_improvement < 1:
        parser.error("Require patience >= 0 and relative improvement in [0, 1)")
    if args.warmup_steps < 0 or args.learning_rate <= 0 or not 0 <= args.min_lr_factor <= 1:
        parser.error("Invalid learning-rate schedule")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    train, val = OccludedDataset(args.train_dir), OccludedDataset(args.val_dir)
    assert train.directory != val.directory
    assert train.manifest["seed"] != val.manifest["seed"], "Validation needs independent trajectories"
    for field in ("teacher_sha256", "camera_intrinsics", "object_keypoints"):
        assert train.manifest[field] == val.manifest[field], f"Dataset mismatch: {field}"
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    checkpoints = output / "pose_estimator"
    checkpoints.mkdir(exist_ok=True)
    module_path = (
        Path(__file__).resolve().parents[2] / "source/viserdex/viserdex/tasks/manipulation/inhand/pose_estimator"
    )
    network = load_module("pose_cnn", module_path / "networks.py").PoseEstimatorNetwork(27).to(args.device)
    transforms = load_module("pose_transforms", module_path / "transforms.py")
    preprocess = transforms.get_transforms(["binary_opening"], args.device)
    augment = transforms.get_transforms(["iso", "motion_blur"], args.device)
    # Materialize LazyLinear before optimizer/checkpoint restoration.
    with torch.no_grad():
        network(torch.zeros(2, 3, 120, 120, device=args.device))
    if args.init_checkpoint:
        network.load_state_dict(
            torch.load(args.init_checkpoint, map_location=args.device, weights_only=True), strict=True
        )
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(args.warmup_steps, max(total_steps - 1, 0))

    def lr_factor(step):
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0), 1)
        return args.min_lr_factor + (1 - args.min_lr_factor) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    loss_fn = torch.nn.HuberLoss(delta=0.05)
    intrinsics = torch.tensor(train.manifest["camera_intrinsics"], device=args.device)
    reference = torch.tensor(train.manifest["object_keypoints"], device=args.device)
    launch = {
        "arguments": vars(args),
        "train_samples": len(train),
        "val_samples": len(val),
        "total_optimizer_steps": total_steps,
        "warmup_optimizer_steps": warmup,
        "teacher_sha256": train.manifest["teacher_sha256"],
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name() if "cuda" in args.device else args.device,
        "initial_weights": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else "ImageNet ResNet34",
        "augmentations": ["3DGS pre-rasterization", "ISO", "motion_blur"],
        "loss": "100 * (Huber(normalized_uv, delta=.05) + Huber(depth_m, delta=.05))",
    }
    if args.resume:
        old = json.loads((output / "launch.json").read_text())
        for field in ("train_samples", "val_samples", "total_optimizer_steps", "teacher_sha256"):
            assert old[field] == launch[field], f"Resume mismatch: {field}"
    else:
        (output / "launch.json").write_text(json.dumps(launch, indent=2))
    step, start_epoch, start_batch, best = 0, 0, 0, float("inf")
    early_stop = EarlyStopping(args.early_stop_patience, args.min_relative_improvement)
    if args.resume:
        state = torch.load(output / "training_state.pt", map_location=args.device, weights_only=False)
        network.load_state_dict(state["network"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, start_batch, best = (state[k] for k in ("step", "epoch", "batch", "best"))
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if "early_stopping" in state:
            early_stop = EarlyStopping(**state["early_stopping"])
    writer = SummaryWriter(str(output / "tensorboard"))
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    started = time.monotonic()

    def record(data):
        row = {"step": step, "elapsed_s": time.monotonic() - started, **data}
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        (output / "progress.json").write_text(json.dumps(row, indent=2))
        print("POSE_DATASET", json.dumps(row), flush=True)
        for key, value in data.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(key, value, step)
        writer.flush()

    def save(epoch, batch, tag):
        state = {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "batch": batch,
            "best": best,
            "early_stopping": asdict(early_stop),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        temporary = output / "training_state.tmp"
        torch.save(state, temporary)
        temporary.replace(output / "training_state.pt")
        torch.save(network.state_dict(), checkpoints / f"cnn_{step}_{tag}.pth")

    loader_options = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )
    val_loader = DataLoader(val, shuffle=False, **loader_options)

    @torch.no_grad()
    def validate():
        network.eval()
        # Keep deterministic opening enabled, as the runtime PoseEstimator does.
        totals, groups = {}, {"low": {}, "medium": {}, "high": {}}
        counts = {name: 0 for name in groups}
        for rgb, target, fraction in val_loader:
            rgb, target = rgb.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            prediction = network(preprocess(rgb.float() / 255))
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite validation prediction")
            metrics = pose_metrics(prediction, target, intrinsics, reference)
            metrics["loss"] = keypoint_loss(prediction, target, loss_fn).expand(len(rgb))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value.sum().item()
            for name, mask in (
                ("low", fraction < 0.25),
                ("medium", (fraction >= 0.25) & (fraction < 0.5)),
                ("high", fraction >= 0.5),
            ):
                counts[name] += int(mask.sum())
                for key, value in metrics.items():
                    groups[name][key] = groups[name].get(key, 0) + value[mask.to(args.device)].sum().item()
        result = {"val/" + key: value / len(val) for key, value in totals.items()}
        for name in groups:
            result[f"val/{name}_count"] = counts[name]
            if counts[name]:
                result.update({f"val/{name}/{key}": value / counts[name] for key, value in groups[name].items()})
        return result

    record({"status": "started", "train_samples": len(train), "val_samples": len(val)})
    completed = False
    stop_reason = "interrupted_or_failed"
    try:
        if args.init_checkpoint:
            baseline = validate()
            best = baseline["val/add_mm"]
            early_stop.update(best)
            torch.save(network.state_dict(), output / "best.pth")
            record({"status": "baseline", "epoch": 0, **baseline})
            save(0, 0, "initial")
        for epoch in range(start_epoch, args.epochs):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            # Explicit epoch permutation lets a resumed run skip exactly the completed batches.
            indices = torch.randperm(len(train), generator=generator).tolist()
            offset = start_batch * args.batch_size if epoch == start_epoch else 0
            loader = DataLoader(train, sampler=indices[offset:], **loader_options)
            network.train()
            loss_sum, count = 0.0, 0
            for batch_index, (rgb, target, _) in enumerate(loader, start=offset // args.batch_size):
                rgb, target = rgb.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
                with torch.no_grad():
                    rgb = augment(preprocess(rgb.float() / 255)).clamp(0, 1)
                optimizer.zero_grad(set_to_none=True)
                prediction = network(rgb)
                loss = keypoint_loss(prediction, target, loss_fn)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at update {step}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                loss_sum += loss.item()
                count += 1
                if step % args.log_interval == 0 or step == 1:
                    record(
                        {
                            "epoch": epoch + 1,
                            "train/loss": loss_sum / count,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
                        }
                    )
                    loss_sum, count = 0.0, 0
                if step % args.save_interval == 0 or stop["requested"]:
                    save(epoch, batch_index + 1, "occluded")
                if stop["requested"]:
                    return
            metrics = validate()
            record({"epoch": epoch + 1, **metrics})
            if metrics["val/add_mm"] < best:
                best = metrics["val/add_mm"]
                torch.save(network.state_dict(), output / "best.pth")
            should_stop = early_stop.update(metrics["val/add_mm"])
            record(
                {
                    "epoch": epoch + 1,
                    "early_stop/bad_epochs": early_stop.bad_epochs,
                    "early_stop/reference_add_mm": early_stop.reference,
                    "best_val_add_mm": best,
                }
            )
            save(epoch + 1, 0, "epoch")
            if should_stop:
                stop_reason = "validation_plateau"
                break
        else:
            stop_reason = "epoch_budget"
        completed = True
    finally:
        (output / "completion.json").write_text(
            json.dumps(
                {
                    "completed": completed,
                    "optimizer_steps": step,
                    "stop_reason": stop_reason,
                    "early_stopping": asdict(early_stop),
                    "best_val_add_mm": best if math.isfinite(best) else None,
                },
                indent=2,
            )
        )
        writer.close()


if __name__ == "__main__":
    main()
