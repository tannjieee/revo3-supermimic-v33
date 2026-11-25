"""Differentiable geometry of the actual rigid hand and object, in SI units.

Geometry is a deterministic readout of the predicted state. It is re-computed
at every imagined step, never independently integrated or teacher-forced.
No changes to the simulator's collision shapes or contact solver are made.
"""

import hashlib
import json
import torch
from torch import nn
from .core import Q, QD, POS, ROT, VEL, OMEGA, quat_apply, quat_mul, quat_conjugate


def geometry_digest(spec):
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class HandObjectGeometry(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.digest = geometry_digest(spec)
        self.body_names = spec["body_names"]
        self.point_names = spec["point_names"]
        self.groups = []
        for i, group in enumerate(spec["levels"]):
            self.groups.append(i)
            for key in ("parent", "joint", "p0", "r0", "p1", "r1", "axis"):
                self.register_buffer(
                    f"{key}_{i}",
                    torch.tensor(
                        group[key],
                        dtype=torch.long
                        if key in ("parent", "joint")
                        else torch.float32,
                    ),
                )
        self.register_buffer(
            "point_body", torch.tensor(spec["point_body"], dtype=torch.long)
        )
        self.register_buffer(
            "point_local", torch.tensor(spec["point_local"], dtype=torch.float32)
        )
        self.object_shape = spec.get("object_shape", "box")
        if self.object_shape == "sphere":
            self.register_buffer("sphere_center", torch.tensor(spec["sphere_center"], dtype=torch.float32))
            self.register_buffer("sphere_radius", torch.tensor(spec["sphere_radius"], dtype=torch.float32))
        elif self.object_shape == "convex_mesh":
            self.register_buffer("convex_planes", torch.tensor(spec["convex_planes"], dtype=torch.float32))
        else:
            self.register_buffer("box_center", torch.tensor(spec["box_center"], dtype=torch.float32))
            self.register_buffer("box_half", torch.tensor(spec["box_half"], dtype=torch.float32))
        self.feature_dim = len(self.point_names) * 10
        object.__setattr__(self, "_compiled_features", None)
        object.__setattr__(self, "_compiled_loss", None)

    def enable_compilation(self):
        # Compile only deterministic geometry. No CUDA graphs, AMP, changed
        # physics, trainable parameters, or altered checkpoint key names.
        object.__setattr__(
            self, "_compiled_features", torch.compile(self.features, fullgraph=True)
        )
        object.__setattr__(
            self, "_compiled_loss", torch.compile(self.loss_components, fullgraph=True)
        )

    def body_kinematics(self, state):
        n = len(state)
        p = state.new_zeros((n, 1, 3))
        v, w = torch.zeros_like(p), torch.zeros_like(p)
        r = torch.cat((p, state.new_ones((n, 1, 1))), -1)
        for i in self.groups:
            parent, joint = getattr(self, f"parent_{i}"), getattr(self, f"joint_{i}")
            pp, rp, vp, wp = p[:, parent], r[:, parent], v[:, parent], w[:, parent]
            p0, r0, p1, r1, axis = (
                getattr(self, f"{key}_{i}") for key in ("p0", "r0", "p1", "r1", "axis")
            )
            rj = quat_mul(rp, r0.expand(n, -1, -1))
            off0 = quat_apply(rp, p0.expand(n, -1, -1))
            pj = pp + off0
            a = state[:, Q][:, joint, None] * 0.5
            rq = torch.cat((axis[None] * torch.sin(a), torch.cos(a)), -1)
            rc = quat_mul(quat_mul(rj, rq), quat_conjugate(r1).expand(n, -1, -1))
            off1 = quat_apply(rc, p1.expand(n, -1, -1))
            wc = (
                wp
                + quat_apply(rj, axis.expand(n, -1, -1)) * state[:, QD][:, joint, None]
            )
            pc = pj - off1
            vc = vp + torch.cross(wp, off0, dim=-1) - torch.cross(wc, off1, dim=-1)
            p, r, v, w = (
                torch.cat((old, new), 1)
                for old, new in ((p, pc), (r, rc), (v, vc), (w, wc))
            )
        return p, r, v, w

    def points(self, state):
        p, r, v, w = self.body_kinematics(state)
        offset = quat_apply(
            r[:, self.point_body], self.point_local.expand(len(state), -1, -1)
        )
        return p[:, self.point_body] + offset, v[:, self.point_body] + torch.cross(
            w[:, self.point_body], offset, dim=-1
        )

    def surface(self, relative):
        """Exact point-to-sphere/box signed distance and outward normal (object frame).

        The shape is extracted from the active USD collision primitive. This is
        an observation feature, not a replacement collision representation.
        Normals at an edge/center use a deterministic face; no normal loss.
        """
        if self.object_shape == "convex_mesh":
            # Exact signed distance to the nearest supporting plane inside the
            # convex solid; outside it is a lower bound on Euclidean distance.
            # This differentiable feature never replaces the physical mesh.
            planes = self.convex_planes
            margins = relative @ planes[:, :3].T + planes[:, 3]
            gap, face = margins.max(dim=-1, keepdim=True)
            return gap, planes[face.squeeze(-1), :3]
        if self.object_shape == "sphere":
            x = relative - self.sphere_center
            length = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            # Center is nondifferentiable; choose +X deterministically and keep gradients finite.
            fallback = torch.zeros_like(x)
            fallback[..., 0] = 1
            normal = torch.where(length > 1e-8, x / length.clamp_min(1e-8), fallback)
            return length - self.sphere_radius, normal
        x = relative - self.box_center
        d = x.abs() - self.box_half
        outside = d.clamp_min(0)
        length = torch.linalg.vector_norm(outside, dim=-1, keepdim=True)
        face = d.argmax(-1)
        sign = torch.where(x >= 0, 1.0, -1.0)
        normal_inside = torch.nn.functional.one_hot(face, 3).to(x.dtype) * sign
        normal_outside = outside * sign / length.clamp_min(1e-8)
        normal = torch.where(length > 0, normal_outside, normal_inside)
        gap = length + d.amax(-1, keepdim=True).clamp_max(0)
        return gap, normal

    def readout(self, state):
        p, v = self.points(state)
        offset = p - state[:, None, POS]
        inverse = quat_conjugate(state[:, ROT])[:, None].expand(
            -1, len(self.point_names), -1
        )
        relative = quat_apply(inverse, offset)
        speed = (
            v
            - state[:, None, VEL]
            - torch.cross(state[:, None, OMEGA].expand_as(offset), offset, dim=-1)
        )
        relative_speed = quat_apply(inverse, speed)
        gap, normal = self.surface(relative)
        return p, v, relative, relative_speed, gap, normal

    def forward(self, state):
        if self._compiled_features is not None:
            return self._compiled_features(state)
        return self.features(state)

    def features(self, state):
        _, _, relative, speed, gap, normal = self.readout(state)
        # Precondition SI quantities before the real-data RunningNorm (eps=.01).
        return torch.cat(
            (relative / 0.05, speed / 0.5, gap / 0.02, normal), -1
        ).flatten(1)

    def supervised_errors(self, prediction, target):
        if self._compiled_loss is not None:
            return self._compiled_loss(prediction, target)
        return self.loss_components(prediction, target)

    def loss_components(self, prediction, target):
        predicted = self.readout(prediction)
        with torch.no_grad():
            actual = self.readout(target)
        # Five group means: hand point p/v, object-relative p/v, surface gap.
        # These SI scales and .5 aggregate weight are Revo3 choices, not paper defaults.
        scales = (0.01, 0.1, 0.01, 0.1, 0.01)
        return torch.stack(
            [
                ((a - b) / scale).square().flatten(1).mean(-1)
                for a, b, scale in zip(predicted[:5], actual[:5], scales)
            ],
            -1,
        )
