"""Deterministic textured RGB-D fixture; does not access a camera.

Ground-truth poses are saved for validation only. The reconstruction command never reads them.
"""

import itertools

import numpy as np

from .common import new_directory, transform_points, write_json


def synthetic(args):
    import cv2

    if args.views < 12 or args.views % 2:
        raise ValueError("Synthetic fixture requires an even number of views, at least 12")
    output = new_directory(args.output)
    for directory in ("images", "depth"):
        (output / directory).mkdir()
    rng = np.random.default_rng(123)
    textures = []
    for _ in range(7):
        texture = rng.integers(40, 230, (512, 512, 3), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (9, 9), 0)
        for j in range(400):
            center = tuple(rng.integers(0, 512, 2).tolist())
            cv2.circle(texture, center, int(rng.integers(2, 13)), rng.integers(0, 256, 3).tolist(), -1)
        textures.append(texture)
    width, height = args.width, args.height
    focal = width * 1.15
    intr = dict(
        width=width, height=height, fx=focal, fy=focal, cx=(width - 1) / 2, cy=(height - 1) / 2, model="PINHOLE"
    )
    v, u = np.mgrid[:height, :width]
    camera_dirs = np.stack(((u - intr["cx"]) / focal, (v - intr["cy"]) / focal, np.ones_like(u)), -1)
    lower, upper = np.array([-0.04, -0.03, -0.025]), np.array([0.04, 0.03, 0.025])
    frames, poses = [], []
    for index in range(args.views):
        theta = 2 * np.pi * (index % (args.views // 2)) / (args.views // 2)
        elevation = np.radians(20 if index < args.views // 2 else 50)
        position = 0.25 * np.array(
            [np.cos(theta) * np.cos(elevation), np.sin(theta) * np.cos(elevation), np.sin(elevation)]
        )
        forward = -position / np.linalg.norm(position)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((right, down, forward))
        pose[:3, 3] = position
        directions = camera_dirs @ pose[:3, :3].T
        safe_dirs = np.where(np.abs(directions) < 1e-9, 1e-9, directions)
        t1, t2 = (lower - position) / safe_dirs, (upper - position) / safe_dirs
        tnear, tfar = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hits = (tfar >= tnear) & (tnear > 0)
        # Textured table, retained for camera tracking and explicitly removed by object crop.
        table_t = (-0.06 - position[2]) / safe_dirs[..., 2]
        table_xyz = position + table_t[..., None] * directions
        table_hit = (table_t > 0) & (np.abs(table_xyz[..., :2]) < 0.25).all(-1)
        depth = np.where(table_hit, table_t, 0.0)
        rgb = np.full((height, width, 3), 235, dtype=np.uint8)
        tu = ((table_xyz[..., 0] + 0.25) / 0.5 * 511).clip(0, 511).astype(int)
        tv = ((table_xyz[..., 1] + 0.25) / 0.5 * 511).clip(0, 511).astype(int)
        rgb[table_hit] = textures[6][tv[table_hit], tu[table_hit]]
        xyz = position + tnear[..., None] * directions
        nearest_face = np.abs(np.concatenate((xyz - lower, xyz - upper), -1)).argmin(-1)
        for face in range(6):
            axes = [axis for axis in range(3) if axis != face % 3]
            pu = (
                ((xyz[..., axes[0]] - lower[axes[0]]) / (upper[axes[0]] - lower[axes[0]]) * 511)
                .clip(0, 511)
                .astype(int)
            )
            pv = (
                ((xyz[..., axes[1]] - lower[axes[1]]) / (upper[axes[1]] - lower[axes[1]]) * 511)
                .clip(0, 511)
                .astype(int)
            )
            selected = hits & (nearest_face == face)
            rgb[selected] = textures[face][pv[selected], pu[selected]]
        depth[hits] = tnear[hits]
        name = f"{index:06d}.png"
        cv2.imwrite(str(output / "images" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / "depth" / name), np.rint(depth / 0.0001).astype(np.uint16))
        frames.append({"image": f"images/{name}", "depth": f"depth/{name}"})
        poses.append(pose.tolist())
    anchor = np.linalg.inv(np.array(poses[0]))
    corners = transform_points(np.array(list(itertools.product(*zip(lower, upper)))), anchor)
    normal = anchor[:3, :3] @ np.array([0, 0, 1.0])
    plane_point = transform_points(np.array([[0, 0, -0.06]]), anchor)[0]
    write_json(
        output / "scan.json",
        {
            "schema_version": 1,
            "completed": True,
            "synthetic": True,
            "mode": "stationary_object_moving_camera",
            "rectified": True,
            "intrinsics": intr,
            "depth_scale_m": 0.0001,
            "depth_convention": "aligned_colour_optical_z",
            "frames": frames,
        },
    )
    write_json(
        output / "ground_truth.json",
        {
            "camera_to_world": poses,
            "bounds_in_first_camera": np.concatenate((corners.min(0) - 0.004, corners.max(0) + 0.004)).tolist(),
            "table_plane_in_first_camera": np.r_[normal, -normal @ plane_point].tolist(),
            "physical_box_extent_m": (upper - lower).tolist(),
        },
    )
    print(f"Wrote {args.views} synthetic RGB-D views to {output}")
