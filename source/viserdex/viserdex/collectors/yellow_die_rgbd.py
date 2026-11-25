"""Isolated simulator capture of yellow-die RGB, first-visible depth and GT labels."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat
from viserdex.distillation.noise import quat_apply, quat_conjugate, quat_mul


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_cameras(cfg, calibration, fixture):
    from scipy.spatial.transform import Rotation
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg, OpenCvPinholeDistortionCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from viserdex.renderer.hikrobot_calibration import world_camera_transform
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils

    record = json.loads(Path(calibration).read_text())
    initial = cfg.scene.robot.init_state
    fixture = Path(fixture).resolve(strict=True)
    # Original handoff fixture is visual-only; do not change policy contacts/dynamics.
    source = fixture.read_text()
    if any(token in source for token in ("PhysicsCollisionAPI", "PhysicsRigidBodyAPI", "physics:")):
        raise ValueError("Capture fixture must be the original visual-only asset")
    cfg.scene.calibration_fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CalibrationFixture",
        spawn=sim_utils.UsdFileCfg(usd_path=str(fixture), semantic_tags=[("class", "calibration_fixture")]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=initial.pos, rot=initial.rot))
    world = world_camera_transform(initial.pos, initial.rot,
                                  {k: np.asarray(v["matrix"]) for k, v in record["transforms"].items()})
    native = np.asarray(record["intrinsics"]["camera_matrix"])
    cfg.scene.object.spawn.semantic_tags = [("class", "rgbd_target")]
    for name, width, height, types in (
        ("rgb_camera", 640, 360, ["rgb", "semantic_segmentation"]),
        ("depth_camera", 848, 480, ["distance_to_image_plane", "semantic_segmentation"]),
    ):
        scale = min(width / 1280, height / 720)
        px, py = (width - 1280 * scale) / 2, (height - 720 * scale) / 2
        affine = np.array([[scale, 0, (scale - 1) / 2 + px],
                           [0, scale, (scale - 1) / 2 + py], [0, 0, 1]])
        k = affine @ native
        spawn = PinholeCameraCfg.from_intrinsic_matrix(
            k.flatten().tolist(), width=width, height=height, focal_length=24., clipping_range=(.001, 3.))
        spawn.distortion = OpenCvPinholeDistortionCfg(
            image_size=(width, height), apply_lens_distortion=True,
            fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2], k1=0., k2=0., k3=0., p1=0., p2=0.)
        setattr(cfg.scene, name, CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name, update_period=0., width=width, height=height,
            data_types=types, update_latest_camera_pose=True, spawn=spawn,
            offset=CameraCfg.OffsetCfg(pos=tuple(world[:3, 3]),
                                      rot=tuple(Rotation.from_matrix(world[:3, :3]).as_quat()), convention="ros"),
            renderer_cfg=NewtonWarpRendererCfg(use_cuda_graph=False, enable_textures=True,
                enable_shadows=False, enable_ambient_lighting=True, colorize_semantic_segmentation=False)))


class YellowDieRGBDCapture:
    def __init__(self, env, calibration, splat_augmentation=None, augmentation_seed=20261101, point_noise_cfg=None, occlusion_audit=False):
        from viserdex.assets.objects import _OBJECT_CFGS
        from viserdex.renderer.gaussian_splat_camera import GaussianSplatCamera

        self.env = env
        self.splat_augmentation = splat_augmentation
        self.augmentation_seed = augmentation_seed
        self.capture_index = 0
        self.point_noise_cfg = point_noise_cfg
        self.point_generator = torch.Generator(device=env.device).manual_seed(augmentation_seed + 1000000)
        calibration_record = json.loads(Path(calibration).read_text())
        self.base_camera = torch.tensor(calibration_record["transforms"]["T_base_camera"]["matrix"],
                                        dtype=torch.float32, device=env.device)
        native = np.asarray(calibration_record["intrinsics"]["camera_matrix"])
        self.expected_intrinsics = []
        for width, height in ((640, 360), (848, 480)):
            scale = min(width / 1280, height / 720)
            px, py = (width - 1280 * scale) / 2, (height - 720 * scale) / 2
            affine = np.array([[scale, 0, (scale-1)/2+px], [0, scale, (scale-1)/2+py], [0, 0, 1]])
            self.expected_intrinsics.append(torch.tensor(affine @ native, device=env.device, dtype=torch.float32))
        self.cameras = [env.scene["rgb_camera"], env.scene["depth_camera"]]
        self.batch = 1
        asset = _OBJECT_CFGS["RoundedDie"]
        loader = SimpleNamespace(cfg=SimpleNamespace(splat_model_path=asset["splat_path"]), _device=env.device)
        GaussianSplatCamera._load_gaussian_splat_model(loader)
        self.splats = loader._splats
        from viserdex.renderer.splat_randomization import ViserDexSplatRandomizer
        self.randomizer = (ViserDexSplatRandomizer(self.splats, splat_augmentation, asset["splat_path"])
                           if splat_augmentation is not None else None)
        asset_metadata = json.loads((Path(asset["usd_path"]).parent / "asset_manifest.json").read_text())
        bounds = np.asarray(asset_metadata["metric_bounds"], dtype=np.float32)
        # Semantic box corners in authored object axes, not points on the rounded surface.
        signs = [(1,1,1),(0,1,1),(1,0,1),(0,0,1),(1,1,0),(0,1,0),(1,0,0),(0,0,0)]
        self.reference = np.vstack((np.zeros((1,3)), [[bounds[x,0],bounds[y,1],bounds[z,2]] for x,y,z in signs])).astype("float32")
        self.contract = dict(object="YellowDie", object_usd_sha256=digest(asset["usd_path"]),
                             splat_sha256=digest(asset["splat_path"]), calibration_sha256=digest(calibration),
                             points_source="scene first-hit target depth, including hand occlusion",
                             rgb_source="3DGS appearance masked by scene first-hit target segmentation",
                             labels="simulator GT only, not outputs of the RSLCube predictor",
                             keypoints="origin + 8 semantic visual bounding-box corners; authored axes: +X face 6, +Z face 5",
                             depth_calibration="virtual co-located depth; not real depth-device calibration")
        from dataclasses import asdict
        self.contract["splat_augmentation"] = (asdict(splat_augmentation) if splat_augmentation else None)
        self.contract["augmentation_seed"] = augmentation_seed
        self.contract["native_augmentation"] = self.randomizer.contract if self.randomizer else None
        self.contract["point_noise"] = point_noise_cfg.contract() if point_noise_cfg else None
        self.contract["point_noise_seed_base"] = augmentation_seed + 2000000
        fixture = Path(env.cfg.scene.calibration_fixture.spawn.usd_path)
        self.contract["fixture"] = dict(path=str(fixture), sha256=digest(fixture),
            texture_sha256=digest(fixture.parent / "apriltag36h11_id0_48mm_quietzone_600dpi.png"),
            placement="robot initial base transform", visual_only=True,
            role="scene appearance and first-hit occlusion; excluded from target RGB and points")
        self.camera_audited = False
        self.auditor = None
        if occlusion_audit:
            from viserdex.collectors.occlusion_audit import OcclusionAudit
            self.auditor = OcclusionAudit(asset["usd_path"], env.cfg.scene.object.spawn.scale, env.device)
            self.contract["occlusion_audit"] = self.auditor.contract
        self.contract["pixel_coordinates"] = "continuous image coordinates; pixel (x,y) centre is (x+0.5,y+0.5)"

    def mask(self, camera, label="rgbd_target", env_ids=None):
        info = camera.data.info["semantic_segmentation"]
        labels = info["idToLabels"]
        ids = [int(k) for k, v in labels.items() if v.get("class") == label]
        if not ids:
            raise ValueError("Renderer did not expose semantic label: " + label)
        image = tensor(camera.data.output["semantic_segmentation"]).squeeze(-1)
        if env_ids is not None:
            image = image[env_ids]
        return torch.isin(image, torch.tensor(ids, device=image.device, dtype=image.dtype))

    @torch.no_grad()
    def capture(self, env_ids=None, refresh_scene=True):
        """Render the actual scene. GT object poses stay on this side of the boundary."""
        from viserdex.renderer.splat_randomization import render_splats

        env = self.env
        batched = env_ids is not None
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        batch = len(env_ids)
        if refresh_scene:
            env.sim.forward()
            env.scene.update(0.)
            for camera in self.cameras:
                camera.update(0., force_recompute=True)
        rgb_camera, depth_camera = self.cameras
        kr, kd = [tensor(c.data.intrinsic_matrices)[env_ids].float() for c in self.cameras]
        camera_pos = tensor(rgb_camera.data.pos_w)[env_ids]
        camera_quat = tensor(rgb_camera.data.quat_w_ros)[env_ids]
        # Both virtual cameras share one optical frame; enforce this assumption.
        torch.testing.assert_close(camera_pos, tensor(depth_camera.data.pos_w)[env_ids], atol=2e-6, rtol=0)
        torch.testing.assert_close(camera_quat, tensor(depth_camera.data.quat_w_ros)[env_ids], atol=2e-6, rtol=0)
        if not self.camera_audited or batched:
            for actual, expected in zip((kr, kd), self.expected_intrinsics, strict=True):
                torch.testing.assert_close(actual, expected[None].expand_as(actual), atol=2e-4, rtol=0)
            robot = env.scene["robot"].data
            base_pos = tensor(robot.root_pos_w)[env_ids]
            base_inv = quat_conjugate(tensor(robot.root_quat_w)[env_ids])
            relative_p = quat_apply(base_inv, camera_pos-base_pos)
            relative_r = matrix_from_quat(quat_mul(base_inv, camera_quat))
            # Float32 world coordinates lose precision far from the grid origin.
            tol = 2e-6 + 4*torch.finfo(base_pos.dtype).eps*float(base_pos.abs().max())
            torch.testing.assert_close(relative_p, self.base_camera[:3, 3].expand_as(relative_p), atol=tol, rtol=0)
            torch.testing.assert_close(relative_r, self.base_camera[:3, :3].expand_as(relative_r), atol=2e-6, rtol=0)
            self.contract["live_camera_calibration_audit"] = "passed: both K matrices and base-to-optical extrinsics"
            self.contract["live_rgb_k"] = kr[0].cpu().tolist()
            self.contract["live_depth_k"] = kd[0].cpu().tolist()
            error = float((relative_p-self.base_camera[:3, 3]).abs().max())
            self.contract['max_camera_translation_error_m'] = max(error, self.contract.get('max_camera_translation_error_m', 0.))
            self.camera_audited = True
        obj = env.scene["object"].data
        camera_inv = quat_conjugate(camera_quat)
        rc = matrix_from_quat(quat_mul(camera_inv, tensor(obj.root_quat_w)[env_ids]))
        tc = quat_apply(camera_inv, tensor(obj.root_pos_w)[env_ids]-camera_pos)
        view = torch.eye(4, device=env.device).repeat(batch, 1, 1)
        view[:, :3, :3], view[:, :3, 3] = rc, tc
        clean = render_splats(self.splats, view, kr)
        rgb_mask = self.mask(rgb_camera, env_ids=env_ids)
        augmentation = [None]*batch
        rendered = clean
        if self.randomizer is not None:
            # Independent native SH draws per environment; bounded image batches.
            rendered = torch.empty_like(clean)
            for i in range(batch):
                perturbed, augmentation[i] = self.randomizer(self.augmentation_seed + self.capture_index+i)
                rendered[i:i+1] = render_splats(perturbed, view[i:i+1], kr[i:i+1])
        first_sample = self.capture_index
        self.capture_index += batch
        # Apply scene first-hit hand occlusion AFTER rendering perturbed splats.
        clean_rgb = (clean * rgb_mask[..., None]).mul(255).round().byte()
        rgb = (rendered * rgb_mask[..., None]).mul(255).round().div(255).permute(0, 3, 1, 2).contiguous()
        depth = tensor(depth_camera.data.output["distance_to_image_plane"])[env_ids].squeeze(-1)
        visible = self.mask(depth_camera, env_ids=env_ids) & torch.isfinite(depth) & (depth > 0) & (depth < 3)
        isolated_depth = self.auditor.render(kd, rc, tc) if self.auditor else None
        audit = [self.auditor.check(depth[i], visible[i], rgb_mask[i], isolated_depth[i])
                 for i in range(batch)] if self.auditor else [None]*batch
        points = torch.zeros(batch, 1024, 3, device=env.device)
        uv = torch.zeros(batch, 1024, 2, device=env.device)
        valid = torch.zeros(batch, 1024, dtype=torch.bool, device=env.device)
        counts = visible.flatten(1).sum(-1)
        for i in range(batch):
            y, x = visible[i].nonzero(as_tuple=True)
            if len(x) == 0:
                continue
            # Match the estimator dataset: unique observed samples, zero padding with valid=False.
            size = min(len(x), 1024)
            take = torch.randperm(len(x), device=env.device, generator=self.point_generator)[:size]
            x, y = x[take], y[take]
            z = depth[i, y, x]
            points[i, :size, 0] = (x+0.5-kd[i, 0, 2])/kd[i, 0, 0]*z
            points[i, :size, 1] = (y+0.5-kd[i, 1, 2])/kd[i, 1, 1]*z
            points[i, :size, 2] = z
            uv[i, :size] = points[i, :size, :2]/z[:, None]*torch.stack((kr[i, 0, 0], kr[i, 1, 1])) + kr[i, :2, 2]
            valid[i, :size] = ((uv[i, :size, 0] >= 0) & (uv[i, :size, 0] < 640)
                               & (uv[i, :size, 1] >= 0) & (uv[i, :size, 1] < 360))
        points[~valid] = 0
        uv[~valid] = 0
        clean_points = points.clone()
        point_depth_noise = torch.zeros_like(valid, dtype=points.dtype)
        point_noise = [None]*batch
        if self.point_noise_cfg is not None:
            from viserdex.collectors.point_noise import add_point_noise
            for i in range(batch):
                points[i], point_depth_noise[i], point_noise[i] = add_point_noise(
                    points[i], valid[i], self.point_noise_cfg, self.augmentation_seed + 2000000 + first_sample+i)
        mesh_rgb = tensor(rgb_camera.data.output["rgb"])[env_ids, ..., :3]
        scene_rgb = torch.where(rgb_mask[..., None], rgb.permute(0,2,3,1).mul(255).byte(), mesh_rgb)
        fixture_rgb = self.mask(rgb_camera, "calibration_fixture", env_ids).flatten(1).sum(-1).cpu().tolist()
        fixture_depth = self.mask(depth_camera, "calibration_fixture", env_ids).flatten(1).sum(-1).cpu().tolist()
        self.last_frame = dict(scene_rgb=scene_rgb, rgb_clean=clean_rgb, env_ids=env_ids,
                               splat_augmentation=augmentation if batched else augmentation[0],
                               occlusion_audit=audit if batched else audit[0], isolated_depth_m=isolated_depth,
                               points_camera_clean_m=clean_points, point_depth_noise_m=point_depth_noise,
                               point_noise=point_noise if batched else point_noise[0],
                               play_raw_rgb=mesh_rgb,
                               visible_fixture_rgb_pixels=fixture_rgb if batched else fixture_rgb[0],
                               visible_fixture_depth_pixels=fixture_depth if batched else fixture_depth[0],
                               depth_m=depth, depth_visible=visible,
                               rgb_visible=rgb_mask, K_rgb=kr, K_depth=kd,
                               camera_pos=camera_pos, camera_quat=camera_quat,
                               object_pos=tensor(obj.root_pos_w)[env_ids].clone(), object_quat=tensor(obj.root_quat_w)[env_ids].clone(),
                               camera_object_rotation=rc, camera_object_translation=tc)
        return rgb, points, uv, valid, counts

