"""Accepted 50 mm / 100 g asset, exact topology scaling, and resume isolation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from viserdex.supermimic.object_variant import VARIANTS, bind_object, object_metadata
from viserdex.supermimic.geometry_asset import build_geometry_spec
from viserdex.assets.revo3_spec import REVO3_COMPACT_USD


def test_visual_and_collision_have_identical_source_topology_and_scale():
    old, new = [Usd.Stage.Open(str(VARIANTS[k])) for k in ('die70', 'die50')]
    for path in ('/Object/Visual/Part_0', '/Object/Collision'):
        a, b = [UsdGeom.Mesh(s.GetPrimAtPath(path)) for s in (old, new)]
        np.testing.assert_array_equal(a.GetFaceVertexIndicesAttr().Get(), b.GetFaceVertexIndicesAttr().Get())
        np.testing.assert_array_equal(a.GetFaceVertexCountsAttr().Get(), b.GetFaceVertexCountsAttr().Get())
        np.testing.assert_allclose(b.GetPointsAttr().Get(), np.asarray(a.GetPointsAttr().Get())*5/7, atol=4e-9)
        np.testing.assert_allclose(np.ptp(b.GetPointsAttr().Get(),axis=0), .05, atol=1e-7)
    assert len(UsdGeom.Mesh(new.GetPrimAtPath('/Object/Collision')).GetPointsAttr().Get()) == 200


def test_mass_and_inertia_match_requested_100g():
    a, b = [build_geometry_spec(REVO3_COMPACT_USD, VARIANTS[k]) for k in ('die70', 'die50')]
    assert b['object_mass'] == pytest.approx(.1, abs=1e-8)
    np.testing.assert_allclose(b['object_inertia'], np.asarray(a['object_inertia'])*(.1/a['object_mass'])*(5/7)**2, rtol=2e-7, atol=1e-12)
    assert np.linalg.eigvalsh(b['object_inertia']).min() > 0
    np.testing.assert_allclose(b['object_com'], np.asarray(a['object_com'])*5/7)
    assert b['object_root_scale'] == [1,1,1]
    assert object_metadata(VARIANTS['die50'])['mass_source'] == 'user_supplied'


def cfg():
    return NS(scene=NS(object=NS(spawn=NS(usd_path=str(VARIANTS['die70']),scale=None))),
              commands=NS(object_pose=NS(goal_pose_visualizer_cfg=NS(markers={'goal':NS(usd_path='old',scale=None)}))))


def test_fresh_default_binds_visual_and_physics_to_58mm():
    c=cfg();bind_object(c)
    assert Path(c.scene.object.spawn.usd_path) == VARIANTS['die58']
    assert c.commands.object_pose.goal_pose_visualizer_cfg.markers['goal'].usd_path == c.scene.object.spawn.usd_path
    assert c.scene.object.spawn.scale is None


def test_saved_70mm_not_silently_changed_and_explicit_mismatch_rejected():
    c=cfg();before=deepcopy(c);bind_object(c,saved=True)
    assert c == before
    with pytest.raises(ValueError,match='fresh run'):bind_object(c,'die50',saved=True)
    assert c == before
    bind_object(c,'die70',saved=True)


def test_manifest_fingerprints_source_and_generated_assets():
    p=VARIANTS['die50'];manifest=json.loads(p.with_name('manifest.json').read_text())
    assert manifest['usd_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest['source_usd_sha256'] == hashlib.sha256(VARIANTS['die70'].with_name('object.usda').read_bytes()).hexdigest()
    assert manifest['accepted_source_validation']['shape_rebuilt'] is False


def test_47g_preserves_50mm_geometry_and_scales_inertia_without_rebinding_saved_100g():
    old, new = [Usd.Stage.Open(str(VARIANTS[k])) for k in ('die50', 'die50_47g')]
    for path in ('/Object/Visual/Part_0', '/Object/Collision'):
        a, b = [UsdGeom.Mesh(s.GetPrimAtPath(path)) for s in (old, new)]
        np.testing.assert_array_equal(a.GetPointsAttr().Get(), b.GetPointsAttr().Get())
        np.testing.assert_array_equal(a.GetFaceVertexIndicesAttr().Get(), b.GetFaceVertexIndicesAttr().Get())
    a,b=[UsdPhysics.MassAPI(s.GetDefaultPrim()) for s in (old,new)]
    # USD authors float32 mass; compare to the representable requested value.
    assert b.GetMassAttr().Get()==float(np.float32(.047))
    np.testing.assert_allclose(b.GetDiagonalInertiaAttr().Get(),np.asarray(a.GetDiagonalInertiaAttr().Get())*.47,rtol=2e-7)
    assert object_metadata(VARIANTS['die50_47g'])['mass_kg']==.047
    c=cfg();bind_object(c,'die50');before=deepcopy(c);bind_object(c,saved=True)
    assert c==before
    with pytest.raises(ValueError,match='fresh run'):bind_object(c,'die50_47g',saved=True)
