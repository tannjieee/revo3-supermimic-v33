"""Evaluate one exact pose checkpoint on a completed Revo3 occlusion dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_pose_dataset import OccludedDataset, load_module
from viserdex.utils.pose_metrics import pose_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Require a positive batch size and nonnegative worker count")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(48)
    dataset = OccludedDataset(args.dataset)
    directory = (
        Path(__file__).resolve().parents[2] / "source/viserdex/viserdex/tasks/manipulation/inhand/pose_estimator"
    )
    model = load_module("pose_eval_cnn", directory / "networks.py").PoseEstimatorNetwork(27).to(args.device).eval()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=args.device), strict=True)
    preprocess = load_module("pose_eval_transforms", directory / "transforms.py").get_transforms(
        ["binary_opening"], args.device
    )
    intrinsics = torch.tensor(dataset.manifest["camera_intrinsics"], device=args.device)
    reference = torch.tensor(dataset.manifest["object_keypoints"], device=args.device)
    samples, occlusions = {}, []
    for rgb, labels, fraction in DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers):
        prediction = model(preprocess(rgb.to(args.device).float() / 255))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite pose prediction")
        for key, value in pose_metrics(prediction, labels.to(args.device), intrinsics, reference).items():
            samples.setdefault(key, []).append(value.cpu())
        occlusions.append(fraction)
    values = {key: torch.cat(rows) for key, rows in samples.items()}
    fraction = torch.cat(occlusions)

    def summarize(mask):
        count = int(mask.sum())
        result = {"count": count}
        if count:
            result["mean"] = {key: float(value[mask].mean()) for key, value in values.items()}
            for quantile in (0.5, 0.95, 0.99):
                result[f"p{int(100 * quantile)}"] = {
                    key: float(value[mask].quantile(quantile))
                    for key, value in values.items()
                    if not key.startswith("success_")
                }
        return result

    result = {
        "completed": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "dataset": str(args.dataset.resolve()),
        "dataset_seed": dataset.manifest["seed"],
        "teacher_sha256": dataset.manifest["teacher_sha256"],
        "preprocessing": ["binary_opening"],
        "stochastic_augmentations": False,
        "geometry_precision": "float64",
        "cnn_matmul_tf32": False,
        "all": summarize(torch.ones(len(dataset), dtype=torch.bool)),
        "occlusion_lt_25pct": summarize(fraction < 0.25),
        "occlusion_25_to_50pct": summarize((fraction >= 0.25) & (fraction < 0.5)),
        "occlusion_ge_50pct": summarize(fraction >= 0.5),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("POSE_EVALUATION", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
