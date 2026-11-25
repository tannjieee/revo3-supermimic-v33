"""Seeded adapter for the unchanged ViserDex DefaultAugmentations chain.

Only SH appearance is augmented. The original geometry, source tensors, and
physics RNG state are preserved. Cluster indices must match this exact PLY.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SplatRandomizationCfg:
    preset: str = "viserdex_default"
    cluster_dir: str = "generated/rgbd_capture/yellow_die_viserdex_clusters_v1/indices"

    def __post_init__(self):
        if self.preset != "viserdex_default":
            raise ValueError("Only the native ViserDex default preset is supported")


class ViserDexSplatRandomizer:
    def __init__(self, source, cfg, splat_path):
        from viserdex.renderer.augmentations.default_augmentations import DefaultAugmentations
        from viserdex.renderer.augmentations import augmentations, default_augmentations

        self.source = source
        self.device = source["means3d"].device
        cluster_dir = Path(cfg.cluster_dir)
        if not cluster_dir.is_absolute():
            cluster_dir = ROOT / cluster_dir
        provenance = json.loads((cluster_dir.parent / "manifest.json").read_text())
        if provenance["splat_sha256"] != digest(splat_path):
            raise ValueError("Cluster indices were generated for a different Gaussian asset")
        self.chain, steps, hashes = [], [], {}
        for original in DefaultAugmentations:
            native = deepcopy(original)
            if not set(native.keys) <= {"sh0", "shN"}:
                raise ValueError("This capture adapter only permits native SH appearance augmentations")
            cluster_name = None
            if getattr(native, "cluster_indices_path", None) is not None:
                path = cluster_dir / Path(native.cluster_indices_path).name
                ids = torch.load(path, map_location="cpu", weights_only=True)
                if ids.shape != (len(source["means3d"]),) or ids.dtype != torch.int64:
                    raise ValueError(f"Invalid cluster index shape or dtype: {path}")
                unique = torch.unique(ids)
                if not torch.equal(unique, torch.arange(len(unique))):
                    raise ValueError(f"Cluster IDs must be contiguous: {path}")
                hashes[path.name] = digest(path)
                if hashes[path.name] != provenance["indices_sha256"][path.name]:
                    raise ValueError(f"Cluster index hash mismatch: {path}")
                native.cluster_indices_path = str(path)
                cluster_name = path.name
            self.chain.append(native.func(native, source, self.device))
            steps.append(dict(type=native.func.__name__, keys=list(native.keys),
                              probability=native.probability, operation=native.noise_cfg.operation,
                              minimum=native.noise_cfg.n_min, maximum=native.noise_cfg.n_max,
                              cluster_indices=cluster_name,
                              cluster_fraction=getattr(native, "cluster_fraction", None),
                              noise_per_cluster=getattr(native, "noise_per_cluster", None),
                              apply_per_key=getattr(native, "apply_per_key", None)))
        self.contract = dict(preset=cfg.preset, steps=steps, cluster_sha256=hashes,
                             default_config_sha256=digest(default_augmentations.__file__),
                             implementation_sha256=digest(augmentations.__file__),
                             splat_sha256=provenance["splat_sha256"],
                             unchanged_parameters=["means3d", "scales", "quats", "opacities"])

    @torch.no_grad()
    def __call__(self, seed):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        result = dict(self.source)
        applied = []
        # Native augmentations use global torch RNG. Fork and restore ONLY this
        # process's CPU RNG and selected CUDA device, keeping physics independent.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            for index, augmentation in enumerate(self.chain):
                before = dict(result)
                result = augmentation(result, num_samples=1)
                applied.append(dict(step=index, triggered_keys=[
                    key for key in augmentation._cfg.keys if result[key] is not before[key]]))
        # Native camera augmentation returns a batch dimension for changed keys.
        for key in ("sh0", "shN"):
            if result[key].ndim == self.source[key].ndim + 1:
                result[key] = result[key].squeeze(0)
        changed = [key for key in ("sh0", "shN") if not torch.equal(result[key], self.source[key])]
        for key in self.contract["unchanged_parameters"]:
            if result[key] is not self.source[key]:
                raise RuntimeError(f"Native augmentation unexpectedly replaced geometry: {key}")
        return result, dict(seed=seed, preset="viserdex_default", steps=applied,
                            changed_parameters=changed,
                            unchanged_parameters=self.contract["unchanged_parameters"])


def render_splats(splats, view, intrinsics):
    from gsplat.rendering import rasterization

    rgb, _, _ = rasterization(
        means=splats["means3d"], quats=splats["quats"], scales=splats["scales"].exp(),
        opacities=splats["opacities"].sigmoid(), colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=view, Ks=intrinsics, width=640, height=360, sh_degree=3, near_plane=.01, far_plane=3.,
        packed=True, absgrad=False, sparse_grad=True, rasterize_mode="antialiased",
        render_mode="RGB", distributed=False)
    return rgb.clamp(0, 1)
