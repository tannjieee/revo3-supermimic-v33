"""Verify released originals and prepare path-local copies without changing learned tensors.

Run with the project's Python environment. Original checkpoints/configs remain
byte-for-byte unchanged; generated copies are excluded from Git.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source/viserdex/viserdex/assets"
OLD_ASSETS = "/home/zhangke/viserDex_revo3/source/viserdex/viserdex/assets"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest():
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Released file checksum mismatch: {record['path']}")
    return manifest


def tensor_fingerprint(value):
    """Include tensor names, dtype, shape and raw bytes in a stable digest."""
    import torch

    digest = hashlib.sha256()

    def visit(item, name):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(f"{name}:{array.dtype}:{tuple(array.shape)}:".encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], name + "/" + str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, name + "/" + str(index))

    visit(value, "root")
    return digest.hexdigest()


def prepare(output):
    import torch
    from viserdex.distillation.environment import SavedEnvironmentBinding
    from viserdex.distillation.teacher import teacher_metadata

    verify_manifest()
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output_dir; refusing to overwrite {output}")
    output.mkdir(parents=True)
    assets = output / "assets"
    for name in ("data", "generated/revo3_compact_visual_palm", "generated/usd/RSLCube"):
        shutil.copytree(ASSETS / name, assets / name, ignore=shutil.ignore_patterns("__pycache__", ".thumbs"))
    # Only asset reference strings change. Geometry, materials and dynamics do not.
    for path in assets.rglob("*"):
        if path.suffix not in (".usda", ".usd") or not path.is_file():
            continue
        raw = path.read_bytes()
        if raw.startswith(b"PXR-USDC"):
            continue
        text = raw.decode("utf-8")

        def reference(match):
            value = match[1]
            if value.startswith(OLD_ASSETS + "/"):
                target = assets / value[len(OLD_ASSETS) + 1:]
                if not target.is_file():
                    raise FileNotFoundError(target)
                return "@" + os.path.relpath(target, path.parent) + "@"
            return match[0]

        relocated = re.sub(r"@([^@\n]+)@", reference, text)
        if relocated != text:
            path.write_text(relocated)

    teacher = output / "teacher"
    shutil.copytree(ROOT / "checkpoints/teacher", teacher)
    student = output / "student"
    student.mkdir()
    for source, dest in ((ROOT / "checkpoints/teacher/params/env.yaml", teacher / "params/env.yaml"),
                         (ROOT / "checkpoints/student/env.yaml", student / "env.yaml")):
        dest.write_text(source.read_text().replace(OLD_ASSETS, str(assets)))
    protocol_path = teacher / "params/teacher_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["environment_sha256"] = sha256(teacher / "params/env.yaml")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    teacher_info = teacher_metadata(teacher / "model_4800.pt", "RSLCube")
    binding = SavedEnvironmentBinding(teacher_info["environment_path"], teacher_info["environment_sha256"])
    student_binding = SavedEnvironmentBinding(student / "env.yaml")
    if binding.contract_sha != student_binding.contract_sha:
        raise ValueError("Relocated teacher/student dynamics contracts differ")
    payload = torch.load(ROOT / "checkpoints/student/student_1000.pt", map_location="cpu", weights_only=True)
    fingerprints = {name: tensor_fingerprint(payload[name]) for name in ("student", "optimizer")}
    metadata = copy.deepcopy(payload["metadata"])
    metadata["teacher"] = teacher_info
    metadata["rollout_environment_sha256"] = sha256(student / "env.yaml")
    metadata["environment_binding"].update(
        source_environment=teacher_info["environment_path"],
        source_environment_sha256=teacher_info["environment_sha256"],
        contract_sha256=binding.contract_sha,
        assets=binding.assets,
    )
    payload["metadata"] = metadata
    destination = student / "student_1000.pt"
    torch.save(payload, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    for name, expected in fingerprints.items():
        if tensor_fingerprint(restored[name]) != expected:
            raise AssertionError(f"Relocation changed {name} tensors")
    pose = output / "pose_estimator"
    shutil.copytree(ROOT / "checkpoints/pose_estimator", pose)
    sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
    from export_student import export

    export(destination, student / "exported")
    report = {
        "format": "viserdex_pretrained_local_paths_v1",
        "original_manifest": str(ROOT / "checkpoints/manifest.json"),
        "original_files_unchanged": True,
        "tensor_fingerprints_unchanged": fingerprints,
        "teacher_checkpoint": str(teacher / "model_4800.pt"),
        "student_checkpoint": str(destination),
        "pose_checkpoint": str(pose / "cnn_6202_final.pth"),
        "teacher_environment_sha256": teacher_info["environment_sha256"],
        "student_environment_sha256": metadata["rollout_environment_sha256"],
        "contract_sha256": binding.contract_sha,
        "assets": binding.assets,
        "runtime_student_sha256": sha256(destination),
        "note": "Runtime student hash changes only because path metadata changes; model/optimizer tensors are identical.",
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    verify_manifest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/pretrained_runtime")
    parser.add_argument("--verify_only", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify_only:
        release = verify_manifest()
        print(f"Verified {len(release['files'])} released files")
    else:
        prepare(arguments.output_dir.expanduser().resolve())
