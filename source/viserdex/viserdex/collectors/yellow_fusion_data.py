"""Offline reader for audited yellow-die RGB-D pairs; no simulation imports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_dataset(directory):
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["completed"] and manifest["training_ready"], "Dataset not ready"
    assert manifest["verified"] == {"train": 20000, "validation": 1000}
    records = {}
    paths, seeds = set(), set()
    split_seeds = {}
    report_hashes = {}
    for shard in manifest["shards"]:
        folder = (root / shard["path"]).resolve(strict=True)
        assert sha256(folder / "manifest.json") == shard["manifest_sha256"]
        meta = json.loads((folder / "manifest.json").read_text())
        report_path = root / shard["validation"]
        report = json.loads(report_path.read_text())
        assert report["passed"] and report["samples"] == shard["samples"] == meta["samples"]
        assert meta["completed"] and meta["mode"] == "policy_rollout_rgbd"
        assert report["calibration_sha256"] == manifest["calibration_sha256"]
        assert meta["capture"]["calibration_sha256"] == manifest["calibration_sha256"]
        assert meta["policy"]["checkpoint_sha256"] == manifest["policy_sha256"]
        assert shard["seed"] == meta["seed"] and shard["seed"] not in seeds
        seeds.add(shard["seed"])
        split_seeds[folder] = (shard["split"], shard["seed"], shard["samples"])
        report_hashes[shard["id"]] = sha256(report_path)
    shard_counts = {folder: 0 for folder in split_seeds}
    for split in ("train", "validation"):
        index = root / f"{split}_index.jsonl"
        assert sha256(index) == manifest["index_sha256"][split]
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        assert len(rows) == manifest["verified"][split]
        for row in rows:
            path = (root / row["pair"]).resolve(strict=True)
            assert path not in paths and row["split"] == split
            paths.add(path)
            info = split_seeds[path.parent.parent]
            assert (row["split"], row["seed"]) == info[:2]
            shard_counts[path.parent.parent] += 1
        records[split] = len(rows)
    assert all(shard_counts[p] == info[2] for p, info in split_seeds.items())
    return dict(root=str(root), manifest_sha256=sha256(root / "manifest.json"),
                index_sha256=manifest["index_sha256"], report_sha256=report_hashes,
                samples=records, calibration_sha256=manifest["calibration_sha256"])


class YellowDataset(Dataset):
    def __init__(self, directory, split):
        self.directory = Path(directory).resolve(strict=True)
        self.split = split
        self.rows = [json.loads(line) for line in
                     (self.directory / f"{split}_index.jsonl").read_text().splitlines()]
        with np.load(self.directory / self.rows[0]["pair"], allow_pickle=False) as data:
            self.reference = data["keypoints_object_m"].astype(np.float32)
        assert self.reference.shape == (9, 3)
        assert np.allclose(np.ptp(self.reference, axis=0), 0.070, atol=1e-5)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.directory / self.rows[index]["pair"], allow_pickle=False) as data:
            # Clean observed first-hit points; never use audit geometry or completed surfaces.
            points = data["points_camera_clean_m"].astype(np.float32)
            rgb = data["rgb" if self.split == "train" else "rgb_clean"]
            uv = data["point_uv_rgb"].astype(np.float32)
            valid = data["point_valid"].astype(bool)
            target = data["keypoints_camera_m"].astype(np.float32)
            assert np.array_equal(data["keypoints_object_m"], self.reference)
            assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8
            assert points.shape == (1024, 3) and uv.shape == (1024, 2)
            assert valid.shape == (1024,) and valid.any()
            assert np.isfinite(points).all() and np.isfinite(uv).all() and np.isfinite(target).all()
            return dict(rgb=torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0,
                        points=torch.from_numpy(points), uv=torch.from_numpy(uv),
                        valid=torch.from_numpy(valid), target=torch.from_numpy(target))
