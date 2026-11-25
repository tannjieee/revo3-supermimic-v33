"""Explicit versioned layouts; the 55 physical coordinates remain geometric XYZW."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StateLayout:
    dim: int
    last_action: slice
    goal_pos: slice
    goal_rot: slice
    time: slice | None
    delay_ready: slice | None = None


LEGACY = StateLayout(173, slice(76, 97), slice(160, 163), slice(163, 167), slice(167, 168))
COMPACT = StateLayout(84, slice(55, 76), slice(76, 79), slice(79, 83), None, slice(83, 84))
REFERENCE = StateLayout(131, slice(55, 76), slice(76, 79), slice(79, 83), None)
STATE_PROTOCOL = 'single_pose_last_action_delayed_pd_no_clock_v2'


def layout(state):
    if state.shape[-1] == REFERENCE.dim:
        return REFERENCE
    if state.shape[-1] == COMPACT.dim:
        return COMPACT
    if state.shape[-1] == LEGACY.dim:
        return LEGACY
    raise ValueError(f'Unsupported full state dimension: {state.shape[-1]}')


def fields():
    return dict(joint_position_geometry_rad=[0,21], joint_velocity_rad_s=[21,42],
                object_position_base_m=[42,45], object_quaternion_xyzw=[45,49],
                object_linear_velocity=[49,52], object_angular_velocity=[52,55],
                last_action_after_tanh=[55,76], goal_position_base_m=[76,79],
                goal_quaternion_xyzw=[79,83], actuator_delay_ready=[83,84])
