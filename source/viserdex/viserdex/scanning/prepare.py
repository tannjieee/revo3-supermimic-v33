"""Explicit object crop, geometric masks, and one shared object-local metric frame."""

import os
from pathlib import Path

import numpy as np

from .common import (
    fingerprint,
    inside_bounds,
    load_scan,
    new_directory,
    read_json,
    transform_points,
    unproject,
    validate_bounds,
    write_json,
)
from .reconstruct import fuse_rgbd


def prepare(args):
    import cv2
    import open3d as o3d

    if args.val_every < 2:
        raise ValueError("--val-every must be at least 2 to retain training views")
    reconstruction = Path(args.reconstruction).resolve()
    rec = read_json(reconstruction / "reconstruction.json")
    if not rec.get("completed"):
        raise ValueError("Reconstruction is incomplete")
    scan_root, scan = load_scan(reconstruction / rec["scan_path"])
    lower, upper = validate_bounds(args.bounds[:3], args.bounds[3:])
    plane = None
    if args.plane:
        plane = np.asarray(args.plane, dtype=float)
        if not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
            raise ValueError("Invalid clipping plane")
        plane /= np.linalg.norm(plane[:3])
    output = new_directory(args.output)
    for directory in ("images", "masks", "depth"):
        (output / directory).mkdir()
    masks, frames = [], []
    for frame in rec["frames"]:
        depth = cv2.imread(str(scan_root / frame["depth"]), cv2.IMREAD_UNCHANGED) * scan["depth_scale_m"]
        xyz, v, u = unproject(depth, scan["intrinsics"])
        world = transform_points(xyz, np.asarray(frame["camera_to_world"]))
        keep = inside_bounds(world, lower, upper)
        if plane is not None:
            keep &= world @ plane[:3] + plane[3] > args.plane_margin
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[v[keep], u[keep]] = 255
        # Fill small stereo holes only; large missing regions remain excluded and are reported.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.uint8(labels == largest) * 255
        if (mask > 0).sum() < args.min_mask_pixels:
            continue
        masks.append(mask > 0)
        frames.append(frame)
    if len(frames) < args.min_views:
        raise ValueError(f"Only {len(frames)} views see this crop; need {args.min_views}. Check bounds in scene.ply")
    mesh, cloud = fuse_rgbd(scan_root, scan, frames, args.voxel_size, rec["depth_max_m"], masks)
    bbox = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    mesh, cloud = mesh.crop(bbox), cloud.crop(bbox)
    if plane is not None and len(mesh.vertices):
        mesh.remove_vertices_by_mask(np.asarray(mesh.vertices) @ plane[:3] + plane[3] <= args.plane_margin)
        cloud = cloud.select_by_index(np.where(np.asarray(cloud.points) @ plane[:3] + plane[3] > args.plane_margin)[0])
    if len(cloud.points) < 100 or len(mesh.triangles) < 100:
        raise ValueError("Object geometry is too sparse; capture closer with valid depth and multiple heights")
    # Isolate the largest connected surface to drop depth speckles. Keep the result open if unseen.
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(labels) != np.argmax(counts))
    mesh.remove_unreferenced_vertices()
    center = (mesh.get_min_bound() + mesh.get_max_bound()) / 2
    object_from_world = np.eye(4)
    object_from_world[:3, 3] = -center
    mesh.transform(object_from_world)
    cloud.transform(object_from_world)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(cloud.points) > args.max_initial_points:
        cloud = cloud.select_by_index(
            np.random.default_rng(42).choice(len(cloud.points), args.max_initial_points, replace=False)
        )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(output / "points.ply"), cloud)
    records = []
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        name = f"{index:06d}.png"
        rgb = cv2.imread(str(scan_root / frame["image"]))
        depth = cv2.imread(str(scan_root / frame["depth"]), cv2.IMREAD_UNCHANGED)
        # Preserve original RGB: training composites arbitrary backgrounds from these masks.
        cv2.imwrite(str(output / "images" / name), rgb)
        cv2.imwrite(str(output / "masks" / name), mask.astype(np.uint8) * 255)
        depth[~mask] = 0
        cv2.imwrite(str(output / "depth" / name), depth)
        pose = object_from_world @ np.asarray(frame["camera_to_world"])
        records.append(
            {
                "image": f"images/{name}",
                "mask": f"masks/{name}",
                "depth": f"depth/{name}",
                "camera_to_world": pose.tolist(),
                "source_image": frame["image"],
                "split": "val" if index % args.val_every == 0 else "train",
                "mask_pixels": int(mask.sum()),
            }
        )
        if index < 6:
            preview = rgb.copy()
            preview[~mask] = (preview[~mask] * 0.15).astype(np.uint8)
            cv2.imwrite(str(output / f"mask_preview_{index:02d}.jpg"), preview)
    summary = {
        "schema_version": 1,
        "completed": True,
        "units": "metres",
        "coordinate_frame": "bbox_centred_first_registered_camera_axes",
        "object_from_reconstruction": object_from_world.tolist(),
        "crop_bounds_m": [lower.tolist(), upper.tolist()],
        "keep_plane_positive": plane.tolist() if plane is not None else None,
        "intrinsics": scan["intrinsics"],
        "depth_scale_m": scan["depth_scale_m"],
        "frames": records,
        "bounds_m": [mesh.get_min_bound().tolist(), mesh.get_max_bound().tolist()],
        "extent_m": (mesh.get_max_bound() - mesh.get_min_bound()).tolist(),
        "mesh_watertight": bool(mesh.is_watertight()),
        "initial_points": len(cloud.points),
        "mesh_sha256": fingerprint(output / "mesh.ply"),
        "points_sha256": fingerprint(output / "points.ply"),
        "reconstruction_path": os.path.relpath(reconstruction, output),
        "scale_evidence": rec.get("scale_evidence", {}),
        "synthetic": scan.get("synthetic", False),
    }
    write_json(output / "dataset.json", summary)
    print(f"Prepared {len(records)} views, {len(cloud.points)} initial Gaussians; extents {summary['extent_m']} m.")


def inspect(args):
    import open3d as o3d

    root = Path(args.reconstruction)
    rec = read_json(root / "reconstruction.json")
    print(f"Frame: {rec['coordinate_frame']}; anchor: {rec['anchor_image']}")
    print(f"Scene XYZ bounds (metres): {rec['scene_bounds_m']}")
    print("Use prepare --bounds xmin ymin zmin xmax ymax zmax. Optional --plane nx ny nz d keeps n.X+d > margin.")
    if args.viewer:
        cloud = o3d.io.read_point_cloud(str(root / "scene.ply"))
        visualizer = o3d.visualization.VisualizerWithEditing()
        visualizer.create_window(window_name="Shift+click crop/plane reference points; Q to print coordinates")
        visualizer.add_geometry(cloud)
        visualizer.run()
        indices = visualizer.get_picked_points()
        visualizer.destroy_window()
        for index in indices:
            print("Picked XYZ (m):", np.asarray(cloud.points)[index].tolist())
