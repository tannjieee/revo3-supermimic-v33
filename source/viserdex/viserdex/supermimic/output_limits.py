"""User-selected joint output limits, separate from the immutable PACE fit.

The native delayed PD enforces effort clipping. MJWarp does not enforce the
declared joint velocity limit; never treat its data buffer as proof of a clamp.
"""
import hashlib
import json
import math
from pathlib import Path

from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

LIMITS_PATH = Path(__file__).resolve().parents[4] / 'config/revo3_joint_output_limits.json'


def load_output_limits(path=LIMITS_PATH):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if (value['schema'] != 'revo3_uniform_joint_output_limits_v1'
            or value['coordinate'] != 'joint_output_after_transmission'
            or value['joint_count'] != len(REVO3_JOINT_NAMES)):
        raise ValueError('Unsupported joint output limit profile')
    rpm, effort = value['velocity_limit_rpm'], value['effort_limit_Nm']
    for parameter in (rpm, effort):
        if type(parameter) not in (int, float) or not math.isfinite(parameter) or parameter <= 0:
            raise ValueError('Joint output limits must be finite and positive')
    return dict(value, velocity_limit_rad_s=rpm * (2 * math.pi / 60),
                velocity_limit_deg_s=rpm * 6, source_sha256=hashlib.sha256(raw).hexdigest())


def _values(value):
    if isinstance(value, dict):
        if set(value) != set(REVO3_JOINT_NAMES):
            raise ValueError('Output limits must cover each of the 21 joints by name')
        return [float(value[name]) for name in REVO3_JOINT_NAMES]
    if type(value) in (float, int):
        return [float(value)] * len(REVO3_JOINT_NAMES)
    return None


def configured_output_limits(cfg, profile='fit_03'):
    """Identify the new profile or the exact historical frozen-fit profile.

Reading a saved environment must keep its original dynamics: legacy profiles
remain readable and receive no additional metadata keys. Partial/mixed changes
are rejected instead of being mislabeled as a validated profile.
    """
    from .identified_hand import load_identified_hand
    fit, _ = load_identified_hand(profile)
    effort = _values(cfg.actuator_effort_limit)
    solver_effort = _values(cfg.joint_effort_limit)
    velocity = _values(cfg.actuator_velocity_limit)
    solver_velocity = _values(cfg.joint_velocity_limit)
    legacy = fit['effective_effort_limit_Nm']
    if effort == legacy and solver_effort == legacy and velocity is None and solver_velocity is None:
        return None
    profile = load_output_limits()
    if (effort != [profile['effort_limit_Nm']] * 21 or solver_effort != effort
            or velocity != [profile['velocity_limit_rad_s']] * 21 or solver_velocity != velocity):
        raise ValueError('Actuator limits differ from both the frozen fit and the declared user profile')
    return profile


def environment_output_limits(env):
    # Lightweight context unit fixtures do not construct a simulation config.
    cfg = getattr(env, 'cfg', None)
    if cfg is None:
        return None
    from .context import env_identified_profile
    return configured_output_limits(cfg.scene.robot.actuators['fingers'], profile=env_identified_profile(env))
