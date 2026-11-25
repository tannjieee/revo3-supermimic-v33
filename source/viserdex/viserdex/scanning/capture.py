"""RealSense capture using SDK depth-to-colour alignment, followed by rectification."""

from datetime import datetime, timezone
import importlib.metadata
import signal
import time

import numpy as np

from .common import intrinsics_matrix, new_directory, write_json


def devices():
    import pyrealsense2 as rs

    result = []
    for device in rs.context().query_devices():
        record = {
            key: device.get_info(getattr(rs.camera_info, key))
            for key in ("name", "serial_number", "firmware_version", "usb_type_descriptor")
            if device.supports(getattr(rs.camera_info, key))
        }
        profiles = set()
        for sensor in device.query_sensors():
            for p in sensor.get_stream_profiles():
                if p.is_video_stream_profile() and p.stream_type() in (rs.stream.color, rs.stream.depth):
                    video = p.as_video_stream_profile()
                    profiles.add((str(p.stream_type()), video.width(), video.height(), p.fps(), str(p.format())))
        record["profiles"] = sorted(profiles)
        result.append(record)
    return result


def capture(args):
    import cv2
    import pyrealsense2 as rs

    available = devices()
    selected = [d for d in available if not args.serial or d["serial_number"] == args.serial]
    if len(selected) != 1:
        raise ValueError("Connect exactly one camera or select --serial; use 'doctor --camera'")
    device_info = selected[0]
    usb2 = device_info.get("usb_type_descriptor", "2").startswith("2")
    fps = args.fps or (15 if usb2 else 30)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(device_info["serial_number"])
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, fps)
    # Resolve before creating an output folder or starting the camera.
    config.resolve(rs.pipeline_wrapper(pipeline))
    output = new_directory(args.output)
    (output / "images").mkdir()
    (output / "depth").mkdir()
    manifest = {
        "schema_version": 1,
        "completed": False,
        "mode": "stationary_object_moving_camera",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device_info,
        "sdk_version": importlib.metadata.version("pyrealsense2"),
        "frames": [],
        "depth_convention": "aligned_colour_optical_z",
        "rectified": True,
    }
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = False
    rejected = {"blur": 0, "depth": 0, "timestamp": 0}
    try:
        profile = pipeline.start(config)
        started = True
        depth_sensor = profile.get_device().first_depth_sensor()
        manifest["depth_scale_m"] = depth_sensor.get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        i = dict(
            width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy, model="PINHOLE"
        )
        manifest["intrinsics"] = i
        manifest["raw_colour_calibration"] = {"model": str(intr.model), "coefficients": list(intr.coeffs)}
        if any(abs(c) > 1e-10 for c in intr.coeffs) and intr.model not in (
            rs.distortion.brown_conrady,
            rs.distortion.modified_brown_conrady,
        ):
            raise ValueError(f"Unsupported nonzero colour distortion: {intr.model}")
        k = intrinsics_matrix(i)
        map_x, map_y = cv2.initUndistortRectifyMap(
            k, np.array(intr.coeffs), None, k, (i["width"], i["height"]), cv2.CV_32FC1
        )
        align = rs.align(rs.stream.color)
        print(
            f"Camera {device_info['serial_number']}, USB {device_info.get('usb_type_descriptor')}, "
            f"{args.width}x{args.height}@{fps}. Warmup {args.warmup}s.",
            flush=True,
        )
        warm_until = time.monotonic() + args.warmup
        last_save, start = -float("inf"), time.monotonic()
        while not stopped and len(manifest["frames"]) < args.frames:
            if time.monotonic() - start > args.duration + args.warmup:
                break
            raw = pipeline.wait_for_frames(10000)
            aligned = align.process(raw)
            colour, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not colour or not depth:
                continue
            rgb = cv2.remap(np.asanyarray(colour.get_data()), map_x, map_y, cv2.INTER_LINEAR)
            z16 = cv2.remap(np.asanyarray(depth.get_data()), map_x, map_y, cv2.INTER_NEAREST)
            if args.preview:
                display = rgb.copy()
                cv2.putText(
                    display,
                    f"Saved {len(manifest['frames'])}/{args.frames}  Q: finish",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("ViserDex scan: stationary object / moving camera", display)
                if cv2.waitKey(1) & 255 in (ord("q"), 27):
                    break
            now = time.monotonic()
            if now < warm_until or now - last_save < args.interval:
                continue
            # Source timestamps must be comparable; do not align asynchronous frames silently.
            dt_ms = abs(colour.get_timestamp() - depth.get_timestamp())
            if colour.get_frame_timestamp_domain() != depth.get_frame_timestamp_domain() or dt_ms > 1500 / fps:
                rejected["timestamp"] += 1
                continue
            sharpness = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            valid_ratio = float(((z16 > 0) & (z16 * manifest["depth_scale_m"] < args.depth_max)).mean())
            if sharpness < args.min_sharpness:
                rejected["blur"] += 1
                continue
            if valid_ratio < args.min_depth_fraction:
                rejected["depth"] += 1
                continue
            name = f"{len(manifest['frames']):06d}.png"
            if not cv2.imwrite(str(output / "images" / name), rgb) or not cv2.imwrite(
                str(output / "depth" / name), z16
            ):
                raise OSError("Could not save capture frame")
            manifest["frames"].append(
                {
                    "image": f"images/{name}",
                    "depth": f"depth/{name}",
                    "colour_timestamp_ms": colour.get_timestamp(),
                    "depth_timestamp_ms": depth.get_timestamp(),
                    "sharpness": sharpness,
                    "valid_depth_fraction": valid_ratio,
                }
            )
            last_save = now
            write_json(output / "scan.json", manifest)
            print(f"Saved {name}: sharpness={sharpness:.1f}, depth={valid_ratio:.1%}", flush=True)
        manifest.update(
            completed=len(manifest["frames"]) >= args.min_frames,
            stop_reason="user_stop" if stopped else "capture_finished",
            rejected=rejected,
        )
        if not manifest["completed"]:
            raise ValueError(f"Only {len(manifest['frames'])} usable frames, need {args.min_frames}; capture retained")
    except BaseException as exc:
        manifest["error"] = str(exc)
        raise
    finally:
        write_json(output / "scan.json", manifest)
        if started:
            pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
