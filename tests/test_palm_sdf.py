"""Check the actual generated palm and the unchanged finger/mechanical model."""

import numpy as np
import pytest

from viserdex.assets.revo3_spec import REVO3_MULTIPHYSICS_USD, REVO3_PALM_SDF_USD


@pytest.fixture(scope="module")
def stages():
    from pxr import Usd

    if not REVO3_PALM_SDF_USD.is_file():
        pytest.skip("Generate the palm SDF asset first")
    return Usd.Stage.Open(str(REVO3_MULTIPHYSICS_USD)), Usd.Stage.Open(str(REVO3_PALM_SDF_USD))


def _active(stage):
    from pxr import Usd, UsdPhysics

    return {
        str(p.GetPath()): p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
    }


def test_finger_collision_geometry_unchanged(stages):
    from pxr import UsdGeom, UsdPhysics
    from viserdex.assets.prepare_revo3_palm_sdf import PALM_COLLIDER

    old, new = map(_active, stages)
    assert len(old) == 82 and len(new) == 78
    assert PALM_COLLIDER in new
    fingers = set(old).intersection(new)
    assert len(fingers) == 77
    for path in fingers:
        a, b = UsdGeom.Mesh(old[path]), UsdGeom.Mesh(new[path])
        np.testing.assert_array_equal(a.GetPointsAttr().Get(), b.GetPointsAttr().Get())
        np.testing.assert_array_equal(a.GetFaceVertexIndicesAttr().Get(), b.GetFaceVertexIndicesAttr().Get())
        assert UsdPhysics.MeshCollisionAPI(old[path]).GetApproximationAttr().Get() == (
            UsdPhysics.MeshCollisionAPI(new[path]).GetApproximationAttr().Get()
        )


def test_mechanics_unchanged(stages):
    from viserdex.assets.prepare_revo3_palm_sdf import mechanical_signature

    assert mechanical_signature(stages[0]) == mechanical_signature(stages[1])


def test_palm_is_one_closed_concave_surface(stages):
    import trimesh
    from pxr import UsdGeom
    from viserdex.assets.prepare_revo3_palm_sdf import PALM_COLLIDER

    p = stages[1].GetPrimAtPath(PALM_COLLIDER)
    m = UsdGeom.Mesh(p)
    mesh = trimesh.Trimesh(
        np.asarray(m.GetPointsAttr().Get()), np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=True
    )
    assert len(mesh.vertices) == 602 and len(mesh.faces) == 1200
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.body_count == 1 and mesh.euler_number == 2
    assert mesh.volume > 0
    # The concavity must survive preprocessing; a hull would make this ratio 1.
    assert mesh.convex_hull.volume / mesh.volume > 1.1
    assert p.GetAttribute("newton:sdfMaxResolution").Get() == 512
    assert not p.GetAttribute("newton:hydroelasticEnabled").Get()


def test_sdf_task_uses_official_newton_contacts():
    import viserdex.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

    cfg = parse_env_cfg("Isaac-ViserDex-Repose-Revo3-SDF", device="cuda:0", num_envs=4)
    assert cfg.scene.robot.spawn.usd_path == str(REVO3_PALM_SDF_USD)
    assert not cfg.sim.physics.solver_cfg.use_mujoco_contacts
    assert cfg.sim.physics.collision_cfg.reduce_contacts
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.actions.joint_pos.joint_names == list(REVO3_JOINT_NAMES)
    assert cfg.decimation == 2 and cfg.sim.dt == 1 / 120
