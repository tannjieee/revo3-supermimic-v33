"""Small-object RGB-D trainer built on gsplat 1.5.3's public rasterizer/MCMC strategy.

Reference: nerfstudio-project/gsplat, v1.5.3 examples/simple_trainer.py (Apache-2.0).
This adapter keeps metric coordinates, adds object masks/depth, and exports the
degree-3 PLY layout consumed by ViserDex. No Isaac Sim is needed for training.
"""

import math
from pathlib import Path
import signal
import time

import numpy as np

from .common import fingerprint, intrinsics_matrix, new_directory, read_json, write_json


def write_splat(path, splats):
    """PLY stores log scales, opacity logits, WXYZ rotations, and channel-major SH."""
    from plyfile import PlyData, PlyElement

    values = {key: value.detach().cpu().numpy() for key, value in splats.items()}
    fields = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{j}" for j in range(3)]
    fields += [f"f_rest_{j}" for j in range(45)] + ["opacity"]
    fields += [f"scale_{j}" for j in range(3)] + [f"rot_{j}" for j in range(4)]
    attributes = np.column_stack(
        (
            values["means"],
            np.zeros_like(values["means"]),
            values["sh0"].reshape(-1, 3),
            values["shN"].transpose(0, 2, 1).reshape(-1, 45),
            values["opacities"],
            values["scales"],
            values["quats"],
        )
    )
    if not np.isfinite(attributes).all():
        raise ValueError("Nonfinite Gaussian attributes")
    array = np.empty(len(attributes), dtype=[(field, "f4") for field in fields])
    for j, field in enumerate(fields):
        array[field] = attributes[:, j]
    PlyData([PlyElement.describe(array, "vertex")], text=False).write(str(path))


def ssim(x, y):
    import torch.nn.functional as functional

    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]

    def avg(t):
        return functional.avg_pool2d(t, 7, 1, 3)

    mx, my = avg(x), avg(y)
    vx, vy, cov = avg(x * x) - mx * mx, avg(y * y) - my * my, avg(x * y) - mx * my
    return (
        ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def train(args):
    import cv2
    import open3d as o3d
    import torch
    from scipy.spatial import cKDTree
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    if not torch.cuda.is_available():
        raise RuntimeError("3DGS training needs a CUDA-enabled PyTorch and NVIDIA GPU")
    root = Path(args.dataset).resolve()
    data = read_json(root / "dataset.json")
    if not data.get("completed") or data.get("units") != "metres":
        raise ValueError("Expected completed metric prepared dataset")
    for name in ("mesh", "points"):
        if data.get(f"{name}_sha256") != fingerprint(root / f"{name}.ply"):
            raise ValueError(f"Prepared {name} changed after preparation")
    output = new_directory(args.output)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda"
    cloud = o3d.io.read_point_cloud(str(root / "points.ply"))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if len(points) < 4 or len(colors) != len(points):
        raise ValueError("Need at least four coloured initial points")
    if len(points) > args.max_gaussians:
        selection = rng.choice(len(points), args.max_gaussians, replace=False)
        points, colors = points[selection], colors[selection]
    distances = cKDTree(points).query(points, k=4)[0][:, 1:]
    initial_scales = np.sqrt(np.mean(distances**2, axis=1)).clip(1e-5)
    n = len(points)
    values = {
        "means": points,
        "scales": np.log(initial_scales)[:, None].repeat(3, 1),
        "quats": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "opacities": np.full(n, math.log(0.1 / 0.9)),
        "sh0": ((colors - 0.5) / 0.28209479177387814)[:, None],
        "shN": np.zeros((n, 15, 3)),
    }
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device)) for k, v in values.items()}
    )
    scene_scale = max(np.linalg.norm(data["extent_m"]) / 2, 0.01)
    rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }
    optimizers = {k: torch.optim.Adam([v], lr=rates[k], eps=1e-15) for k, v in splats.items()}
    strategy = MCMCStrategy(
        cap_max=args.max_gaussians,
        refine_start_iter=min(500, args.iterations // 5),
        refine_stop_iter=int(args.iterations * 0.8),
        refine_every=100,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state()
    intr = data["intrinsics"]
    width, height = intr["width"] // args.downsample, intr["height"] // args.downsample
    if min(width, height) < 16:
        raise ValueError("Downsampled images must be at least 16 pixels on each side")
    k = intrinsics_matrix(intr)
    # cv2.resize uses half-pixel centres; keep projection consistent for odd dimensions too.
    sx, sy = width / intr["width"], height / intr["height"]
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2], k[1, 2] = (k[0, 2] + 0.5) * sx - 0.5, (k[1, 2] + 0.5) * sy - 0.5
    kt = torch.tensor(k, dtype=torch.float32, device=device)[None]
    # Only cache on CPU, keeping GPU memory bounded for hundreds of capture frames.
    samples = []
    for frame in data["frames"]:
        rgb = cv2.cvtColor(cv2.imread(str(root / frame["image"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(root / frame["mask"]), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(root / frame["depth"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
        depth = (
            cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            * data["depth_scale_m"]
        )
        samples.append(
            {
                "rgb": torch.from_numpy(rgb),
                "mask": torch.from_numpy(mask[..., None]),
                "depth": torch.from_numpy(depth),
                "split": frame["split"],
                "view": torch.tensor(np.linalg.inv(frame["camera_to_world"]), dtype=torch.float32),
            }
        )
    train_ids = [i for i, sample in enumerate(samples) if sample["split"] == "train"]
    val_ids = [i for i, sample in enumerate(samples) if sample["split"] == "val"]
    if not train_ids or not val_ids:
        raise ValueError("Both training and held-out views are required")
    manifest_hash = fingerprint(root / "dataset.json")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    config.update(
        dataset_manifest_sha256=manifest_hash,
        torch_version=torch.__version__,
        units="metres",
        strategy="gsplat.MCMCStrategy",
        sh_degree=3,
    )
    write_json(output / "config.json", config)

    def render(index, background, degree):
        sample = samples[index]
        rgb, mask, depth = (sample[k].to(device) for k in ("rgb", "mask", "depth"))
        rendered, alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"].exp(),
            opacities=splats["opacities"].sigmoid(),
            colors=torch.cat((splats["sh0"], splats["shN"]), 1),
            viewmats=sample["view"].to(device)[None],
            Ks=kt,
            width=width,
            height=height,
            sh_degree=degree,
            packed=False,
            near_plane=0.01,
            far_plane=10.0,
            render_mode="RGB+ED",
            backgrounds=background[None],
        )
        target = rgb * mask + background * (1 - mask)
        return rendered[0], alpha[0], target, mask, depth, info

    def evaluate(tag, save_previews=False):
        metrics = []
        with torch.no_grad():
            for j, index in enumerate(val_ids):
                image, alpha, target, mask, _, _ = render(index, torch.ones(3, device=device), 3)
                error = (image[..., :3] - target).square()
                foreground_mse = (error * mask).sum() / (3 * mask.sum()).clamp_min(1)
                intersection = ((alpha > 0.5) & (mask > 0.5)).sum()
                union = ((alpha > 0.5) | (mask > 0.5)).sum().clamp_min(1)
                metrics.append(
                    {
                        "psnr": float(-10 * torch.log10(error.mean().clamp_min(1e-10))),
                        "foreground_psnr": float(-10 * torch.log10(foreground_mse.clamp_min(1e-10))),
                        "mask_iou": float(intersection / union),
                    }
                )
                if save_previews and j < 6:
                    pair = torch.cat((target, image[..., :3].clamp(0, 1)), dim=1)
                    pixels = (pair.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(str(output / f"{tag}_{j:02d}.png"), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        return {"views": len(metrics), **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}}

    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    step = 0
    start = time.monotonic()
    completion = {"completed": False}
    try:
        initial_metrics = evaluate("initial")
        for step in range(1, args.iterations + 1):
            degree = min(3, step // max(1, args.iterations // 5))
            background = torch.rand(3, device=device)
            image, alpha, target, mask, depth, info = render(int(rng.choice(train_ids)), background, degree)
            abs_error = (image[..., :3] - target).abs()
            rgb_loss = (abs_error * mask).sum() / (3 * mask.sum()).clamp_min(1)
            rgb_loss += 0.1 * (abs_error * (1 - mask)).sum() / (3 * (1 - mask).sum()).clamp_min(1)
            valid = (depth > 0) & (mask[..., 0] > 0.5)
            depth_loss = (
                (image[..., 3][valid] - depth[valid]).abs().mean() / scene_scale if valid.any() else image.sum() * 0
            )
            loss = 0.8 * rgb_loss + 0.2 * (1 - ssim(image[..., :3], target)) + 0.1 * (alpha - mask).abs().mean()
            loss = loss + args.depth_weight * depth_loss + 0.001 * splats["opacities"].sigmoid().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            lr = rates["means"] * 0.01 ** (step / args.iterations)
            optimizers["means"].param_groups[0]["lr"] = lr
            strategy.step_post_backward(splats, optimizers, state, step, info, lr=lr)
            with torch.no_grad():
                splats["scales"].clamp_(math.log(1e-5), math.log(scene_scale))
                splats["quats"].copy_(torch.nn.functional.normalize(splats["quats"], dim=-1))
            if step == 1 or step % args.log_every == 0 or stopped or step == args.iterations:
                record = dict(
                    step=step,
                    loss=float(loss.detach()),
                    rgb_loss=float(rgb_loss.detach()),
                    depth_loss=float(depth_loss.detach()),
                    gaussians=len(splats["means"]),
                    elapsed_seconds=time.monotonic() - start,
                )
                write_json(output / "progress.json", record)
                with (output / "metrics.jsonl").open("a") as stream:
                    import json

                    stream.write(json.dumps(record) + "\n")
                print(record, flush=True)
            if step % args.save_every == 0 or stopped or step == args.iterations:
                checkpoint = {
                    "step": step,
                    "splats": splats.state_dict(),
                    "config": config,
                    "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                }
                torch.save(checkpoint, output / "checkpoint.pt.tmp")
                (output / "checkpoint.pt.tmp").replace(output / "checkpoint.pt")
            if stopped:
                break
        if stopped:
            completion.update(stop_reason="signal", step=step)
            return
        # Object-only export; eliminate low opacity / escaped splats and evaluate the EXACT export.
        lower, upper = torch.tensor(data["bounds_m"], device=device)
        keep = (splats["opacities"].sigmoid() > 0.01) & (
            (splats["means"] >= lower - args.export_margin) & (splats["means"] <= upper + args.export_margin)
        ).all(-1)
        if keep.sum() < 10:
            raise ValueError("Training produced fewer than ten usable object Gaussians")
        splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v[keep].detach()) for k, v in splats.items()})
        final_metrics = evaluate("validation", save_previews=True)
        write_splat(output / "splat.ply", splats)
        write_json(
            output / "evaluation.json",
            {
                "initial": initial_metrics,
                "exported": final_metrics,
                "note": "Held-out RGB views; geometry/poses were reconstructed from all views",
            },
        )
        completion.update(
            completed=True,
            stop_reason="max_iterations",
            step=step,
            dataset_manifest_sha256=manifest_hash,
            splat_sha256=fingerprint(output / "splat.ply"),
            gaussian_count=int(keep.sum()),
            units="metres",
            sh_degree=3,
            exported_metrics=final_metrics,
            elapsed_seconds=time.monotonic() - start,
        )
    except BaseException as exc:
        completion.update(error=str(exc), step=step, stop_reason="error")
        raise
    finally:
        write_json(output / "completion.json", completion)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
