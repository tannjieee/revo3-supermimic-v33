"""Fail-fast physical validity checks and bounded, device-resident evidence.

These checks do not modify simulator state or select easy training samples. A
failure stops collection before the offending transition can enter replay.
"""
import math
import torch


MECHANICAL_LIMIT_TOLERANCE_RAD = math.radians(5.0)
TRACE_CONTROL_STEPS = 40


def audit_compiled_joint_limits(solver, joint_names, mechanical_limits):
    """Read the compiled MJWarp limits, not just the authored USD/Lab buffers.

    Intended for startup/isolated diagnostics; never called per control step.
    All replicated worlds must retain the authored range and enable each limit.
    """
    import numpy as np
    model = solver.mj_model
    compiled_names = [model.joint(i).name for i in range(model.njnt)]
    indices = []
    for name in joint_names:
        matches = [i for i, compiled in enumerate(compiled_names)
                   if compiled == name or compiled.endswith('_' + name)]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely map compiled joint {name}: {matches}')
        indices.append(matches[0])
    expected = np.asarray(mechanical_limits, dtype=float)
    ranges = solver.mjw_model.jnt_range.numpy()
    limited = solver.mjw_model.jnt_limited.numpy()
    if ranges.ndim == 2:
        ranges = ranges[None]
    if limited.ndim == 1:
        limited = limited[None]
    selected = ranges[:, indices]
    if not np.all(limited[:, indices]) or not np.allclose(selected, expected[None], atol=2e-6, rtol=0):
        raise ValueError('Compiled MJWarp mechanical limits differ from the authored asset')
    result = dict(joint_names=list(joint_names), compiled_joint_indices=indices,
                  compiled_joint_names=[compiled_names[i] for i in indices],
                  authored_range_rad=expected.tolist(), all_compiled_limits_enabled=True,
                  maximum_range_error_rad=float(np.max(np.abs(selected - expected[None]))))
    for name in ('jnt_solref', 'jnt_solimp', 'jnt_margin'):
        array = getattr(solver.mjw_model, name).numpy()
        # MJWarp model fields have a leading world/broadcast axis.
        selected = array[:, indices]
        result[name] = dict(minimum=np.min(selected, axis=0).tolist(), maximum=np.max(selected, axis=0).tolist())
    return result


class PhysicsHealthMonitor:
    def __init__(self, mechanical_limits, joint_names, control_period_s,
                 trace_steps=TRACE_CONTROL_STEPS):
        self.limits = mechanical_limits.detach().clone()
        if self.limits.ndim != 2 or self.limits.shape != (len(joint_names), 2):
            raise ValueError('Mechanical health limits must have one lower/upper pair per joint')
        if not torch.isfinite(self.limits).all() or not (self.limits[:, 0] < self.limits[:, 1]).all():
            raise ValueError('Mechanical health limits must be finite and ordered')
        if type(trace_steps) is not int or trace_steps < 1:
            raise ValueError('Physical trace length must be a positive integer')
        self.joint_names = list(joint_names)
        self.capacity, self.count, self.cursor = trace_steps, 0, 0
        self.control_period_s = float(control_period_s)
        self.buffers = {}
        self.completed = []

    def manifest(self):
        return dict(protocol='mechanical_limit_fail_fast_v1',
                    mechanical_limit_tolerance_rad=MECHANICAL_LIMIT_TOLERANCE_RAD,
                    mechanical_limit_tolerance_deg=5.0,
                    threshold_reason='Gross mechanical-stop violation; permits small soft-constraint overshoot',
                    limit_source='USD authored geometric mechanical limits, not policy encoder limits',
                    trace_control_steps=self.capacity,
                    trace_duration_s=self.capacity * self.control_period_s,
                    response='Fatal before replay insertion; no clipping, filtering, or silent reset',
                    velocity_is_diagnostic_only=True)

    def _write(self, name, value):
        if name not in self.buffers:
            self.buffers[name] = torch.zeros((self.capacity, *value.shape), dtype=value.dtype, device=value.device)
        self.buffers[name][self.cursor].copy_(value.detach())

    @torch.no_grad()
    def begin(self, state, action):
        self.cursor = self.count % self.capacity
        self._write('state', state)
        self._write('action', action)
        if not self.completed:
            self.completed = [False] * self.capacity
        self.completed[self.cursor] = False
        self.count += 1

    @torch.no_grad()
    def finish(self, physical, target, computed_torque=None, applied_torque=None):
        self._write('next_physical', physical)
        self._write('encoder_target', target)
        for name, value in (('computed_torque', computed_torque), ('applied_torque', applied_torque)):
            if value is not None:
                self._write(name, value)
        self.completed[self.cursor] = True

    @torch.no_grad()
    def inspect(self, physical, phase):
        n = len(self.joint_names)
        q = physical[:, :n]
        excess = torch.maximum(self.limits[:, 0] - q, q - self.limits[:, 1]).clamp_min(0)
        nonfinite = ~torch.isfinite(physical).all(-1)
        invalid_limit = (excess > MECHANICAL_LIMIT_TOLERANCE_RAD).any(-1)
        invalid = nonfinite | invalid_limit
        if not bool(invalid.any()):
            return None
        worlds = invalid.nonzero(as_tuple=True)[0]
        safe_excess = torch.nan_to_num(excess, nan=0., posinf=float('inf'), neginf=0.)
        worst_flat = int(safe_excess.argmax())
        world, joint = divmod(worst_flat, n)
        flags = []
        if bool(invalid_limit.any()):
            flags.append('MECHANICAL_LIMIT')
        if bool(nonfinite.any()):
            flags.append('NONFINITE_PHYSICAL_STATE')
        def finite_float(value):
            value = float(value)
            return value if math.isfinite(value) else None
        return dict(kind='invalid_physical_state', flags=flags, phase=phase,
                    affected_worlds=worlds.cpu().tolist(), control_step=self.count,
                    mechanical_limit_tolerance_rad=MECHANICAL_LIMIT_TOLERANCE_RAD,
                    mechanical_limit_tolerance_deg=5.0,
                    worst_world=world, worst_joint=self.joint_names[joint],
                    joint_angle_rad=finite_float(q[world, joint]),
                    mechanical_lower_rad=float(self.limits[joint, 0]),
                    mechanical_upper_rad=float(self.limits[joint, 1]),
                    maximum_mechanical_limit_violation_rad=finite_float(safe_excess.max()),
                    maximum_absolute_joint_velocity_rad_s=finite_float(physical[:, n:2*n].abs().max()),
                    health_contract=self.manifest())

    def snapshot(self, worlds=None):
        length = min(self.count, self.capacity)
        indices = [(self.count - length + offset) % self.capacity for offset in range(length)]
        if worlds is None:
            worlds = list(range(self.buffers['state'].shape[1])) if self.buffers else []
        result = dict(manifest=self.manifest(), world_ids=list(worlds),
                      control_steps=list(range(self.count - length + 1, self.count + 1)),
                      complete=[self.completed[i] for i in indices], joint_names=self.joint_names,
                      mechanical_limits_rad=self.limits.cpu().clone())
        for name, value in self.buffers.items():
            selected = value[indices][:, worlds].detach().cpu().clone()
            # An exception before post-step capture must not expose a stale slot
            # from an earlier wrap as though it belonged to the failing step.
            if name not in ('state', 'action'):
                for row, complete in enumerate(result['complete']):
                    if not complete:
                        selected[row].fill_(float('nan'))
            result[name] = selected
        return result
