"""58 mm package preserves authored geometry/inertia and isolates old runs."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from viserdex.supermimic.object_variant import VARIANTS, bind_object, object_metadata
from test_die50_asset import cfg


def test_58mm_preserves_package_and_newton_hull_budget():
    path=VARIANTS['die58']
    manifest=json.loads(path.with_name('manifest.json').read_text())
    source,generated=[Usd.Stage.Open(str(p)) for p in (manifest['source'],path)]
    for name in ('/Object/Visual/Part_0','/Object/Collision'):
        a,b=[UsdGeom.Mesh(s.GetPrimAtPath(name)) for s in (source,generated)]
        np.testing.assert_array_equal(a.GetPointsAttr().Get(),b.GetPointsAttr().Get())
        np.testing.assert_array_equal(a.GetFaceVertexIndicesAttr().Get(),b.GetFaceVertexIndicesAttr().Get())
        np.testing.assert_allclose(np.ptp(b.GetPointsAttr().Get(),axis=0),.058,atol=1e-7)
    a,b=[UsdPhysics.MassAPI(s.GetDefaultPrim()) for s in (source,generated)]
    for getter in ('GetMassAttr','GetCenterOfMassAttr','GetDiagonalInertiaAttr','GetPrincipalAxesAttr'):
        assert getattr(a,getter)().Get()==getattr(b,getter)().Get()
    assert generated.GetPrimAtPath('/Object/Collision').GetAttribute('newton:maxHullVertices').Get()==200
    metadata=object_metadata(path)
    assert metadata['mass_is_estimated'] is True
    assert metadata['mass_kg']==pytest.approx(.073362112)


def test_saved_50mm_cannot_silently_become_58mm():
    c=cfg();bind_object(c,'die50_47g');before=deepcopy(c)
    bind_object(c,saved=True);assert c==before
    with pytest.raises(ValueError,match='fresh run'):
        bind_object(c,'die58',saved=True)
    assert c==before
    bind_object(c,'die58');assert Path(c.scene.object.spawn.usd_path)==VARIANTS['die58']
