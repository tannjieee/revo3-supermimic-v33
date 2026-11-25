"""Prepare native ViserDex SH augmentation indices without starting a simulator.

Reuse the pure functions in the original clustering script, omitting its
module-level AppLauncher. Native settings: means3d K=128; SH0 Lab color,
automatic K up to 32; KMeans max_iter=100, n_init=1, random_state=42.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/viserdex"))
from viserdex.utils.load_utils import load_splat_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--splat", type=Path, default=ROOT / "assets/rounded_die/gaussians.ply")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "generated/rgbd_capture/yellow_die_viserdex_clusters_v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace existing indices")
    torch.set_num_threads(2)
    native = ROOT / "scripts/preprocessing/cluster_splat.py"
    module = ast.parse(native.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"rgb_to_lab", "cluster_"}]
    if len(functions) != 2:
        raise RuntimeError("Native clustering function layout changed")
    namespace = dict(np=np, torch=torch, KMeans=KMeans)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(native), "exec"), namespace)
    splats = load_splat_file(str(args.splat), device="cpu")
    out = args.output / "indices"
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(completed=False, splat_sha256=digest(args.splat),
                    native_preprocessing_sha256=digest(native), indices_sha256={}, clusters={})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for key, maximum, automatic in [("means3d", 128, False), ("sh0", 32, True)]:
        values = splats[key]
        if key == "sh0":
            rgb = (values * 0.28209479177387814 + 0.5).clamp(0, 1)
            values = torch.tensor(namespace["rgb_to_lab"](rgb.numpy()), dtype=values.dtype)
        indices, count = namespace["cluster_"](values, maximum, automatic, maximum)
        path = out / f"splat_clustered_{key}_indices.pt"
        torch.save(torch.tensor(indices, dtype=torch.int64), path)
        manifest["indices_sha256"][path.name] = digest(path)
        manifest["clusters"][key] = dict(count=int(count), automatic=automatic, maximum=maximum)
    manifest["completed"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
