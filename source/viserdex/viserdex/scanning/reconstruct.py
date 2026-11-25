"""Calibrated COLMAP SfM, robust RGB-D scale recovery, and Open3D TSDF fusion."""

import os
from pathlib import Path

import numpy as np

from .common import load_scan, metric_scale, new_directory, write_json


def fuse_rgbd(scan_root, scan, frames, voxel_size, depth_max, masks=None):
    import cv2
    import open3d as o3d

    i = scan["intrinsics"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(i["width"], i["height"], i["fx"], i["fy"], i["cx"], i["cy"])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=4 * voxel_size, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for index, frame in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(str(scan_root / frame["image"])), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(scan_root / frame["depth"]), cv2.IMREAD_UNCHANGED)
        if masks is not None:
            depth = depth.copy()
            depth[~masks[index]] = 0
        image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(depth),
            depth_scale=1 / scan["depth_scale_m"],
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(image, intrinsic, np.linalg.inv(np.asarray(frame["camera_to_world"])))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh, volume.extract_point_cloud()


def reconstruct(args):
    import cv2
    import open3d as o3d
    import pycolmap

    scan_root, scan = load_scan(args.scan)
    output = new_directory(args.output)
    status = {"completed": False, "stage": "sfm", "pycolmap_version": pycolmap.__version__}
    write_json(output / "status.json", status)
    try:
        database = output / "database.db"
        i = scan["intrinsics"]
        reader = pycolmap.ImageReaderOptions(
            camera_model="PINHOLE", camera_params=",".join(str(i[k]) for k in ("fx", "fy", "cx", "cy"))
        )
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = args.threads
        extraction.sift.max_num_features = args.max_features
        pycolmap.extract_features(
            str(database),
            str(scan_root / "images"),
            image_names=[Path(f["image"]).name for f in scan["frames"]],
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="PINHOLE",
            reader_options=reader,
            extraction_options=extraction,
            device=pycolmap.Device.cpu,
        )
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = args.threads
        matcher = args.matcher
        if matcher == "auto":
            matcher = "exhaustive" if len(scan["frames"]) <= 300 else "sequential"
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(str(database), matching_options=matching, device=pycolmap.Device.cpu)
        else:
            pycolmap.match_sequential(
                str(database),
                matching_options=matching,
                pairing_options=pycolmap.SequentialPairingOptions(
                    overlap=args.overlap, quadratic_overlap=True, loop_detection=False
                ),
                device=pycolmap.Device.cpu,
            )
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = args.threads
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
        options.min_model_size = min(10, len(scan["frames"]))
        (output / "sparse").mkdir()
        models = pycolmap.incremental_mapping(
            str(database), str(scan_root / "images"), str(output / "sparse"), options=options
        )
        if not models:
            raise ValueError("COLMAP could not reconstruct a model; add texture, overlap and translated viewpoints")
        model_id, model = max(models.items(), key=lambda item: item[1].num_reg_images())
        fraction = model.num_reg_images() / len(scan["frames"])
        if fraction < args.min_registered_fraction:
            raise ValueError(f"Only {fraction:.1%} of images registered in the largest model; inspect sparse/")
        images = {image.name: image for image in model.images.values() if image.has_pose}
        frames, measured, predicted, image_scales = [], [], [], []
        for frame in scan["frames"]:
            image = images.get(Path(frame["image"]).name)
            if image is None:
                continue
            w2c = np.eye(4)
            w2c[:3] = image.cam_from_world().matrix()
            depth = cv2.imread(str(scan_root / frame["depth"]), cv2.IMREAD_UNCHANGED) * scan["depth_scale_m"]
            local_measured, local_predicted = [], []
            for point in image.points2D:
                if not point.has_point3D():
                    continue
                u, v = np.rint(point.xy).astype(int)
                if not (1 <= u < i["width"] - 1 and 1 <= v < i["height"] - 1):
                    continue
                patch = depth[v - 1 : v + 2, u - 1 : u + 2]
                good = patch[(patch > args.depth_min) & (patch < args.depth_max)]
                # Depth discontinuities / RGB-depth disocclusions corrupt scale recovery.
                if len(good) < 7 or np.ptp(good) > 0.02:
                    continue
                xyz = model.points3D[point.point3D_id].xyz
                z = (w2c[:3, :3] @ xyz + w2c[:3, 3])[2]
                if z > 0:
                    local_measured.append(float(np.median(good)))
                    local_predicted.append(float(z))
            measured.extend(local_measured)
            predicted.extend(local_predicted)
            if len(local_measured) >= 20:
                image_scales.append(float(np.median(np.array(local_measured) / local_predicted)))
            frames.append({**frame, "camera_to_world": np.linalg.inv(w2c).tolist()})
        scale, evidence = metric_scale(measured, predicted)
        if len(image_scales) < 3:
            raise ValueError("Metric scale needs depth evidence from at least three registered images")
        evidence["per_image_relative_p90_error"] = float(np.percentile(np.abs(np.array(image_scales) / scale - 1), 90))
        if evidence["per_image_relative_p90_error"] > 0.15:
            raise ValueError("Scale varies across views: check depth alignment, blur or nonrigid/moving scene")
        # Anchor every result to the first REGISTERED rectified optical camera: x right,y down,z forward.
        anchor = np.array(frames[0]["camera_to_world"])
        anchor[:3, 3] *= scale
        world_from_scaled_sfm = np.linalg.inv(anchor)
        for frame in frames:
            pose = np.array(frame["camera_to_world"])
            pose[:3, 3] *= scale
            frame["camera_to_world"] = (world_from_scaled_sfm @ pose).tolist()
        summary = {
            "schema_version": 1,
            "completed": False,
            "scan_path": os.path.relpath(scan_root, output),
            "coordinate_frame": "first_registered_rectified_camera_opencv_metres",
            "anchor_image": frames[0]["image"],
            "metres_per_sfm_unit": scale,
            "world_from_scaled_sfm": world_from_scaled_sfm.tolist(),
            "scale_evidence": evidence,
            "selected_model": int(model_id),
            "registered_fraction": fraction,
            "matcher": matcher,
            "frames": frames,
            "intrinsics": i,
            "depth_scale_m": scan["depth_scale_m"],
            "depth_max_m": args.depth_max,
            "voxel_size_m": args.voxel_size,
        }
        write_json(output / "status.json", {**status, "stage": "tsdf", "scale_evidence": evidence})
        mesh, cloud = fuse_rgbd(scan_root, scan, frames, args.voxel_size, args.depth_max)
        if len(cloud.points) < 100:
            raise ValueError("Too few TSDF points: check metric poses and depth range")
        o3d.io.write_point_cloud(str(output / "scene.ply"), cloud)
        o3d.io.write_triangle_mesh(str(output / "scene_mesh.ply"), mesh)
        summary.update(completed=True, scene_bounds_m=[cloud.get_min_bound().tolist(), cloud.get_max_bound().tolist()])
        write_json(output / "reconstruction.json", summary)
        status.update(completed=True, stage="completed", registered_images=len(frames))
    except BaseException as exc:
        status.update(stage="failed", error=str(exc))
        raise
    finally:
        write_json(output / "status.json", status)
    print(f"Reconstructed {len(frames)} images; metric scale={scale:.6g}. Crop scene.ply before training.")
