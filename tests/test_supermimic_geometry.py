"""Independent analytic, finite-difference and training tests for geometry WM."""

import copy
import math
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from viserdex.supermimic.geometry import HandObjectGeometry
from viserdex.supermimic.networks import Dynamics
from viserdex.supermimic.trainer import Trainer, frozen
from viserdex.supermimic.core import POS, ROT, VEL, OMEGA, Q, QD, ema_target
from test_supermimic import config, fixture, make_replay


def spec():
    identity = [0, 0, 0, 1]
    return dict(
        version=1,
        body_names=["root", "link1", "link2"],
        levels=[
            dict(
                parent=[0],
                joint=[0],
                p0=[[0, 0, 0]],
                r0=[identity],
                p1=[[0, 0, 0]],
                r1=[identity],
                axis=[[0, 0, 1]],
            ),
            dict(
                parent=[1],
                joint=[1],
                p0=[[1, 0, 0]],
                r0=[identity],
                p1=[[0, 0, 0]],
                r1=[identity],
                axis=[[0, 0, 1]],
            ),
        ],
        point_body=[2],
        point_local=[[1, 0, 0]],
        point_names=["tip"],
        box_center=[0, 0, 0],
        box_half=[0.1, 0.2, 0.3],
    )


def test_two_joint_analytic_fk_and_velocity():
    g = HandObjectGeometry(spec())
    state, _ = fixture(1)
    state[:, 0] = math.pi / 2
    state[:, 21] = 2
    state[:, 22] = 3
    p, v = g.points(state)
    torch.testing.assert_close(
        p[0, 0], torch.tensor([0.0, 2.0, 0.0]), atol=3e-7, rtol=0
    )
    torch.testing.assert_close(
        v[0, 0], torch.tensor([-7.0, 0.0, 0.0]), atol=2e-6, rtol=0
    )


def test_fk_velocity_matches_pose_finite_difference():
    torch.manual_seed(4)
    g = HandObjectGeometry(spec()).double()
    state, _ = fixture(5)
    state = state.double()
    state[:, :2] = torch.randn(5, 2, dtype=torch.double)
    state[:, 21:23] = torch.randn(5, 2, dtype=torch.double)
    plus, minus = state.clone(), state.clone()
    eps = 1e-6
    plus[:, Q] += eps * state[:, QD]
    minus[:, Q] -= eps * state[:, QD]
    finite = (g.points(plus)[0] - g.points(minus)[0]) / (2 * eps)
    torch.testing.assert_close(g.points(state)[1], finite, atol=1e-8, rtol=1e-6)


def test_nonzero_child_joint_frame_is_respected():
    description = spec()
    description["levels"][1]["p1"] = [[0.2, 0, 0]]
    g = HandObjectGeometry(description)
    state, _ = fixture(1)
    state[:, 1] = math.pi / 2
    p, _ = g.points(state)
    torch.testing.assert_close(
        p[0, 0], torch.tensor([1.0, 0.8, 0.0]), atol=3e-7, rtol=0
    )


def test_relative_velocity_includes_object_rotation():
    g = HandObjectGeometry(spec())
    state, _ = fixture(1)
    state[:, POS] = 0
    state[:, VEL] = torch.tensor([0.3, 0.4, 0.5])
    state[:, OMEGA] = torch.tensor([0.0, 0.0, 2.0])
    readout = g.readout(state)
    torch.testing.assert_close(readout[3][0, 0], torch.tensor([-0.3, -4.4, -0.5]))


def test_exact_box_distances_normals_and_finite_gradients():
    g = HandObjectGeometry(spec())
    points = torch.tensor(
        [[[0.4, 0.6, 0.3], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]], requires_grad=True
    )
    gap, norm = g.surface(points)
    torch.testing.assert_close(gap.flatten(), torch.tensor([0.5, -0.1, 0.0]))
    torch.testing.assert_close(norm[0, 0], torch.tensor([0.6, 0.8, 0.0]))
    (gap.sum() + norm.square().sum()).backward()
    assert torch.isfinite(points.grad).all()


def test_spatial_loss_zero_and_gradients_reach_joint_and_object_state():
    g = HandObjectGeometry(spec())
    target, _ = fixture(2)
    assert not g.supervised_errors(target, target).any()
    pred = target.clone()
    pred[:, 0] = 0.1
    pred[:, 42] += 0.01
    pred.requires_grad_()
    g.supervised_errors(pred, target).mean().backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, 0].abs().sum() > 0 and pred.grad[:, 42].abs().sum() > 0


def test_geometry_wm_action_gradient_through_frozen_model():
    cfg = config(geometry_enabled=True, grouped_physical_loss=True)
    g = Dynamics(2, cfg, spec())
    state, limits = fixture()
    with torch.no_grad():
        g.net[-1].weight.normal_(0, 0.01)
    action = torch.zeros(6, 21, requires_grad=True)
    with frozen(g):
        g(state, ema_target(state, action, limits), torch.ones(6, 2))[
            :, POS
        ].sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    assert all(p.grad is None for p in g.parameters())
    assert g.input_norm.mean.numel() == 57 + 21 + 2 + 10


def test_geometry_training_checkpoint_and_microbatch_parity(tmp_path):
    cfg = config(geometry_enabled=True, grouped_physical_loss=True)
    replay, state, limits, noise = make_replay(cfg)
    inverse = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1)
    a = Trainer(
        cfg, noise, torch.ones(6, 2), limits, inverse, "cpu", geometry_spec=spec()
    )
    a.update_real_statistics(
        state, torch.zeros(6, 21), state[:, :55], replay.data["history"][0, :, 0]
    )
    b = copy.deepcopy(a)
    b.cfg = copy.deepcopy(cfg)
    b.cfg.microbatch = 6
    rng = torch.get_rng_state()
    metrics = a.wm_update(replay)
    torch.set_rng_state(rng)
    b.wm_update(replay)
    for p, q in zip(a.wm.parameters(), b.wm.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    assert "wm_loss_relative_velocity" in metrics
    a.policy_update(replay)
    metadata = dict(
        environment_contract="a",
        joint_names=[],
        context_schema={},
        observation_protocol="v1",
    )
    a.save(tmp_path / "geometry.pt", metadata)
    payload = torch.load(tmp_path / "geometry.pt", weights_only=False)
    assert payload["version"] == 17
    a.restore(payload, metadata)
    payload["geometry_spec"] = copy.deepcopy(payload["geometry_spec"])
    payload["geometry_spec"]["box_half"][0] *= 2
    with pytest.raises(ValueError, match="geometry contract"):
        a.restore(payload, metadata)


def test_actual_usd_tree_against_independent_homogeneous_transforms():
    description = json.loads(
        (Path(__file__).parent / "data/revo3_geometry_v1.json").read_text()
    )
    g = HandObjectGeometry(description).double()
    torch.manual_seed(147)
    state, _ = fixture(7)
    state = state.double()
    state[:, Q] = torch.randn(7, 21, dtype=torch.double) * 0.7
    state[:, QD] = torch.randn(7, 21, dtype=torch.double)

    def matrix(q):
        x, y, z, w = np.array(q) / np.linalg.norm(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def transform(p, q):
        out = np.eye(4)
        out[:3, :3] = matrix(q)
        out[:3, 3] = p
        return out

    expected = []
    for row in state.numpy():
        transforms = [np.eye(4)]
        for level in description["levels"]:
            for i, j in enumerate(level["joint"]):
                a = row[j] / 2
                q = [*(np.array(level["axis"][i]) * np.sin(a)), np.cos(a)]
                transforms.append(
                    transforms[level["parent"][i]]
                    @ transform(level["p0"][i], level["r0"][i])
                    @ transform([0, 0, 0], q)
                    @ np.linalg.inv(transform(level["p1"][i], level["r1"][i]))
                )
        expected.append(
            [
                (transforms[i] @ np.array([*local, 1]))[:3]
                for i, local in zip(
                    description["point_body"], description["point_local"]
                )
            ]
        )
    torch.testing.assert_close(
        g.points(state)[0], torch.tensor(np.array(expected)), atol=2e-7, rtol=1e-5
    )
    plus, minus = state.clone(), state.clone()
    eps = 1e-6
    plus[:, Q] += eps * state[:, QD]
    minus[:, Q] -= eps * state[:, QD]
    torch.testing.assert_close(
        g.points(state)[1],
        (g.points(plus)[0] - g.points(minus)[0]) / (2 * eps),
        atol=2e-7,
        rtol=1e-5,
    )


def test_actual_geometry_feature_action_directional_derivative():
    description = json.loads(
        (Path(__file__).parent / "data/revo3_geometry_v1.json").read_text()
    )
    g = HandObjectGeometry(description).double()
    state, _ = fixture(3)
    state = state.double()
    torch.manual_seed(77)
    state[:, :42] = torch.randn(3, 42, dtype=torch.double) * 0.3
    state.requires_grad_()
    weights = torch.randn(3, g.feature_dim, dtype=torch.double)
    direction = torch.zeros_like(state)
    direction[:, :42] = torch.randn(3, 42, dtype=torch.double)
    direction /= direction.norm()
    f = lambda x: (g(x) * weights).sum()
    grad = torch.autograd.grad(f(state), state)[0]
    eps = 1e-6
    finite = (f(state + eps * direction) - f(state - eps * direction)) / (2 * eps)
    torch.testing.assert_close((grad * direction).sum(), finite, atol=1e-5, rtol=1e-5)


def test_asset_extraction_includes_rigid_root_scale(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    from viserdex.supermimic.geometry_asset import build_geometry_spec

    path = tmp_path / "scaled_box.usda"
    stage = Usd.Stage.CreateNew(str(path))
    body = UsdGeom.Xform.Define(stage, "/object")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    body.AddScaleOp().Set((0.9, 0.9, 0.9))
    collider = UsdGeom.Cube.Define(stage, "/object/collision")
    collider.GetSizeAttr().Set(2.0)
    collider.AddScaleOp().Set((0.0325, 0.0325, 0.0325))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    stage.GetRootLayer().Save()
    hand = (
        Path(__file__).parents[1]
        / "source/viserdex/viserdex/assets/generated/revo3_compact_visual_palm/revo3_right_compact_v4/revo3_right_compact_v4.usda"
    )
    result = build_geometry_spec(hand, path)
    np.testing.assert_allclose(result["box_half"], [0.02925] * 3, atol=1e-8)
