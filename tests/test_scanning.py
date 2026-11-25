"""Geometry/protocol tests for the optional scanning pipeline (no camera or Isaac Sim)."""

import numpy as np
import pytest

from viserdex.assets.scanned_objects import load_scanned_objects
from viserdex.scanning.common import (
    inside_bounds,
    metric_scale,
    transform_points,
    unproject,
    validate_bounds,
    write_json,
)


def test_metric_scale_rejects_depth_edges_and_recovers_real_units():
    rng = np.random.default_rng(5)
    sfm = rng.uniform(2, 10, 1000)
    depth = sfm * 0.043 * rng.normal(1, 0.003, 1000)
    depth[:80] *= 3
    scale, evidence = metric_scale(depth, sfm)
    assert scale == pytest.approx(0.043, rel=0.001)
    assert evidence["inliers"] == 920
    with pytest.raises(ValueError, match="Inconsistent"):
        metric_scale(sfm * rng.uniform(0.01, 0.2, 1000), sfm)
    with pytest.raises(ValueError, match="Only"):
        metric_scale([0.3], [1])


def test_optical_depth_and_object_transform_preserve_reprojection():
    intr = dict(fx=400.0, fy=420.0, cx=1.0, cy=1.0)
    depth = np.full((3, 3), 0.4)
    xyz, v, u = unproject(depth, intr)
    assert np.allclose(xyz[:, 2], 0.4)  # optical Z, not radial distance
    c2w = np.array([[0, 0, 1, 0.3], [1, 0, 0, -0.2], [0, 1, 0, 0.1], [0, 0, 0, 1.0]])
    o2w = np.eye(4)
    o2w[:3, 3] = [0.1, 0.05, -0.2]
    points_object = transform_points(transform_points(xyz, c2w), np.linalg.inv(o2w))
    camera_to_object = np.linalg.inv(o2w) @ c2w
    recovered = transform_points(points_object, np.linalg.inv(camera_to_object))
    assert np.allclose(recovered, xyz)
    assert np.allclose(recovered[:, 0] / recovered[:, 2] * intr["fx"] + intr["cx"], u)
    assert np.allclose(recovered[:, 1] / recovered[:, 2] * intr["fy"] + intr["cy"], v)


def test_crop_bounds_validate_and_do_not_invert_axes():
    lo, hi = validate_bounds([-0.1, -0.2, 0.3], [0.1, 0.2, 0.6])
    assert inside_bounds(np.array([[0, 0, 0.4], [0, 0, 0], [0.2, 0, 0.4]]), lo, hi).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        validate_bounds([0, 0, 0], [1, -1, 1])
    with pytest.raises(ValueError):
        validate_bounds([0, 0, 0], [1, np.nan, 1])


def test_gaussian_ply_roundtrip_matches_existing_project_loader(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("plyfile")
    from viserdex.scanning.train import write_splat
    from viserdex.scanning.assets import validate_splat
    from viserdex.utils.load_utils import load_splat_file

    gen = torch.Generator().manual_seed(3)
    n = 20
    original = {
        "means": torch.randn(n, 3, generator=gen) * 0.03,
        "scales": torch.full((n, 3), -6.0),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1),
        "opacities": torch.randn(n, generator=gen),
        "sh0": torch.randn(n, 1, 3, generator=gen),
        "shN": torch.randn(n, 15, 3, generator=gen),
    }
    file = tmp_path / "splat.ply"
    write_splat(file, original)
    validate_splat(file)
    actual = load_splat_file(str(file))
    for key, value in original.items():
        assert torch.equal(actual["means3d" if key == "means" else key], value)


def test_registry_is_portable_and_rejects_shadowing_or_path_escape(tmp_path):
    folder = tmp_path / "MyObject"
    folder.mkdir()
    paths = dict(usd_path="object.usda", mesh_path="mesh.ply", splat_path="splat.ply", points_path="points.pt")
    for file in paths.values():
        (folder / file).touch()
    manifest = {"schema_version": 1, "name": "MyObject", **paths}
    write_json(folder / "object.json", manifest)
    result = load_scanned_objects(tmp_path)
    assert result["MyObject"]["mesh_path"] == str(folder / "mesh.ply")
    with pytest.raises(ValueError, match="shadow"):
        load_scanned_objects(tmp_path, reserved_names=["MyObject"])
    (tmp_path / "external.ply").touch()
    write_json(folder / "object.json", {**manifest, "mesh_path": "../external.ply"})
    with pytest.raises(ValueError, match="external"):
        load_scanned_objects(tmp_path)


def test_saved_usd_keeps_mesh_units_and_rigid_body(tmp_path):
    o3d = pytest.importorskip("open3d")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    from viserdex.scanning.assets import create_usd

    mesh = o3d.geometry.TriangleMesh.create_box(0.08, 0.06, 0.05)
    mesh.translate([-0.04, -0.03, -0.025])
    file = tmp_path / "object.usda"
    create_usd(mesh, file, 0.071)
    stage = Usd.Stage.Open(str(file))
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdPhysics.MassAPI(stage.GetDefaultPrim()).GetMassAttr().Get() == pytest.approx(0.071)
    points = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/Object/geometry")).GetPointsAttr().Get())
    assert np.allclose(points.max(0) - points.min(0), [0.08, 0.06, 0.05])


def test_scan_rejects_incomplete_data_and_existing_directories(tmp_path):
    from viserdex.scanning.common import load_scan, new_directory

    write_json(tmp_path / "scan.json", {"schema_version": 1, "completed": False})
    with pytest.raises(ValueError, match="completed"):
        load_scan(tmp_path)
    with pytest.raises(FileExistsError):
        new_directory(tmp_path)


def test_import_requires_matching_training_geometry(tmp_path):
    pytest.importorskip("open3d")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from types import SimpleNamespace
    from viserdex.scanning.assets import import_asset

    dataset, training = tmp_path / "data", tmp_path / "training"
    write_json(dataset / "dataset.json", {"completed": True})
    write_json(training / "completion.json", {"completed": True, "dataset_manifest_sha256": "wrong"})
    with pytest.raises(ValueError, match="different prepared"):
        import_asset(SimpleNamespace(name="Test", dataset=dataset, training=training))


def test_metric_capture_rejects_bad_scale_and_duplicate_frames(tmp_path):
    from viserdex.scanning.common import load_scan

    frames = []
    for index in range(3):
        image, depth = f"rgb_{index}.png", f"depth_{index}.png"
        (tmp_path / image).touch()
        (tmp_path / depth).touch()
        frames.append({"image": image, "depth": depth})
    data = {
        "schema_version": 1,
        "completed": True,
        "rectified": True,
        "intrinsics": dict(width=640, height=480, fx=400, fy=400, cx=320, cy=240, model="PINHOLE"),
        "depth_scale_m": -0.001,
        "frames": frames,
    }
    write_json(tmp_path / "scan.json", data)
    with pytest.raises(ValueError, match="Depth scale"):
        load_scan(tmp_path)
    data["depth_scale_m"] = 0.001
    data["frames"][2] = data["frames"][0]
    write_json(tmp_path / "scan.json", data)
    with pytest.raises(ValueError, match="Duplicate"):
        load_scan(tmp_path)


def test_pipeline_does_not_import_after_reconstruction_failure(tmp_path):
    pytest.importorskip("open3d")
    pytest.importorskip("pycolmap")
    from viserdex.scanning.cli import build_parser
    from viserdex.scanning.common import read_json

    file = tmp_path / "pipeline.json"
    write_json(
        file,
        {
            "reconstruct": {"scan": str(tmp_path / "absent"), "output": str(tmp_path / "reconstruction")},
            "import": {
                "dataset": "absent",
                "training": "absent",
                "name": "NeverPublished",
                "mass_kg": 0.05,
                "assets_dir": str(tmp_path / "assets"),
            },
        },
    )
    args = build_parser().parse_args(["process", "--config", str(file)])
    with pytest.raises(FileNotFoundError):
        args.func(args)
    assert not (tmp_path / "assets").exists()
    status = read_json(tmp_path / "pipeline.status.json")
    assert not status["completed"] and status["stage"] == "reconstruct"
