"""Stage-oriented command line interface. Imports heavy libraries only when needed."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .common import read_json, write_json


def doctor(args):
    report = {"python": sys.executable, "packages": {}}
    modules = {"opencv-python": "cv2", "usd-core": "pxr.Usd", "scikit-learn": "sklearn"}
    import_errors = {}
    required = ["numpy", "opencv-python", "pyrealsense2"]
    if not args.capture_only:
        required += ["pycolmap", "open3d", "torch", "gsplat", "scipy", "plyfile", "usd-core", "scikit-learn"]
    for package in required:
        try:
            report["packages"][package] = importlib.metadata.version(package)
            importlib.import_module(modules.get(package, package))
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
        except Exception as exc:
            import_errors[package] = str(exc)
    report["import_errors"] = import_errors
    if args.camera:
        from .capture import devices

        report["cameras"] = devices()
    if args.cuda:
        import torch

        report["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
        if torch.cuda.is_available():
            report["cuda"].update(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
        if args.rasterizer:
            from gsplat import rasterization

            means = torch.tensor([[0.0, 0.0, 0.5]], device="cuda", requires_grad=True)
            out, _, _ = rasterization(
                means,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
                torch.full((1, 3), 0.01, device="cuda"),
                torch.ones(1, device="cuda"),
                torch.ones(1, 3, device="cuda"),
                torch.eye(4, device="cuda")[None],
                torch.tensor([[[100.0, 0, 16], [0, 100.0, 16], [0, 0, 1.0]]], device="cuda"),
                32,
                32,
            )
            out.sum().backward()
            report["cuda"]["rasterizer_backward"] = bool(torch.isfinite(means.grad).all() and out.max() > 0)
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)
    if import_errors or any(value is None for value in report["packages"].values()):
        raise RuntimeError("Missing/broken packages; run scripts/setup_scan.sh")


def dispatch(module, function):
    def run(args):
        return getattr(importlib.import_module(f"viserdex.scanning.{module}"), function)(args)

    return run


def process(args):
    """Run saved explicit configuration; completed stages are checked, never overwritten."""
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    allowed = ("reconstruct", "prepare", "train", "import")
    if not config or any(key not in allowed for key in config):
        raise ValueError(f"Config keys must be stages {allowed}; capture is deliberately separate")
    pipeline = config_path.with_suffix(".status.json")
    for stage in allowed:
        if stage not in config:
            continue
        argv = [stage]
        options = config[stage]
        for key, value in options.items():
            if isinstance(value, bool):
                if value:
                    argv.append("--" + key.replace("_", "-"))
            elif value is not None:
                argv += ["--" + key.replace("_", "-")]
                argv += [str(v) for v in value] if isinstance(value, list) else [str(value)]
        write_json(pipeline, {"completed": False, "stage": stage, "arguments": argv})
        try:
            parsed = build_parser().parse_args(argv)
            parsed.func(parsed)
            if stage == "train" and not read_json(Path(parsed.output) / "completion.json").get("completed"):
                raise RuntimeError("Training stopped before budget; import was not started")
        except BaseException as exc:
            write_json(pipeline, {"completed": False, "stage": stage, "error": str(exc)})
            raise
    write_json(pipeline, {"completed": True, "stage": "completed"})


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def positive_float(text):
    import math

    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="D435i RGB-D → metric reconstruction → object 3DGS → ViserDex")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="Check dependencies/device without starting a camera stream")
    for flag in ("camera", "cuda", "rasterizer", "capture-only"):
        p.add_argument("--" + flag, action="store_true")
    p.add_argument("--output", type=Path)
    p.set_defaults(func=doctor)
    p = sub.add_parser("capture", help="Capture a fixed object while moving the camera")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--serial")
    p.add_argument("--width", type=positive_int, default=640)
    p.add_argument("--height", type=positive_int, default=480)
    p.add_argument("--fps", type=positive_int)
    p.add_argument("--frames", type=positive_int, default=240)
    p.add_argument("--min-frames", type=positive_int, default=30)
    p.add_argument("--duration", type=positive_float, default=180)
    p.add_argument("--warmup", type=positive_float, default=3)
    p.add_argument("--interval", type=positive_float, default=0.5)
    p.add_argument("--min-sharpness", type=positive_float, default=35)
    p.add_argument("--min-depth-fraction", type=positive_float, default=0.15)
    p.add_argument("--depth-max", type=positive_float, default=1.5)
    p.add_argument("--preview", action="store_true")
    p.set_defaults(func=dispatch("capture", "capture"))
    p = sub.add_parser("reconstruct", help="COLMAP poses, metric depth alignment and TSDF scene")
    p.add_argument("--scan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=positive_int, default=8)
    p.add_argument("--max-features", type=positive_int, default=8192)
    p.add_argument(
        "--matcher",
        choices=("auto", "sequential", "exhaustive"),
        default="auto",
        help="auto: exhaustive for <=300 views, sequential for longer captures",
    )
    p.add_argument("--overlap", type=positive_int, default=15)
    p.add_argument("--min-registered-fraction", type=positive_float, default=0.8)
    p.add_argument("--depth-min", type=positive_float, default=0.1)
    p.add_argument("--depth-max", type=positive_float, default=1.5)
    p.add_argument("--voxel-size", type=positive_float, default=0.002)
    p.set_defaults(func=dispatch("reconstruct", "reconstruct"))
    p = sub.add_parser("inspect", help="Inspect metric scene and optionally pick crop coordinates")
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--viewer", action="store_true")
    p.set_defaults(func=dispatch("prepare", "inspect"))
    p = sub.add_parser("prepare", help="Crop object, fuse its mesh, and generate training masks")
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--bounds", type=float, nargs=6, required=True, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX")
    )
    p.add_argument("--plane", type=float, nargs=4, help="Keep n.X+d > margin, in reconstruction coordinates")
    p.add_argument("--plane-margin", type=positive_float, default=0.003)
    p.add_argument("--voxel-size", type=positive_float, default=0.0015)
    p.add_argument("--min-mask-pixels", type=positive_int, default=300)
    p.add_argument("--min-views", type=positive_int, default=12)
    p.add_argument("--max-initial-points", type=positive_int, default=15000)
    p.add_argument("--val-every", type=positive_int, default=8)
    p.set_defaults(func=dispatch("prepare", "prepare"))
    p = sub.add_parser("train", help="Train object-only degree-3 Gaussians on CUDA")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=positive_int, default=7000)
    p.add_argument("--max-gaussians", type=positive_int, default=50000)
    p.add_argument("--downsample", type=positive_int, default=1)
    p.add_argument("--threads", type=positive_int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=positive_int, default=100)
    p.add_argument("--save-every", type=positive_int, default=1000)
    p.add_argument("--depth-weight", type=positive_float, default=0.01)
    p.add_argument("--export-margin", type=positive_float, default=0.005)
    p.set_defaults(func=dispatch("train", "train"))
    p = sub.add_parser("import", help="Publish mesh/USD/PLY/keypoints/clusters as one object bundle")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mass-kg", type=positive_float, required=True)
    default_assets = Path(__file__).resolve().parents[1] / "assets/data/scanned"
    p.add_argument("--assets-dir", type=Path, default=default_assets)
    p.set_defaults(func=dispatch("assets", "import_asset"))
    p = sub.add_parser("synthetic", help="Generate textured RGB-D test data without accessing the camera")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--views", type=positive_int, default=48)
    p.add_argument("--width", type=positive_int, default=480)
    p.add_argument("--height", type=positive_int, default=360)
    p.set_defaults(func=dispatch("synthetic", "synthetic"))
    p = sub.add_parser("smoke", help="Run full synthetic RGB-D → COLMAP → 3DGS → import validation")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=positive_int, default=600)
    p.set_defaults(func=dispatch("smoke", "smoke"))
    p = sub.add_parser("process", help="Execute stages from an explicit JSON configuration")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=process)
    return parser


def main():
    args = build_parser().parse_args()
    if getattr(args, "rasterizer", False):
        args.cuda = True
    args.func(args)
