"""Frozen hardware identification profiles, verified by SHA256 before use (SI units).

Supports both the legacy 2026-11-02 PACE ``fit_03`` bundle and the 2026-11-24
``revo3-sysid`` Fresh-50Hz 21-DOF closed-loop identification bundles.
"""
from pathlib import Path
import hashlib
import json
import math
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

DATA_DIR = Path(__file__).parent / 'data'
ROOT = DATA_DIR / 'fit_03'
SUPPORTED_PROFILES = ('fit_03', 'revo3_sysid_fresh50hz_v1', 'revo3_sysid_20261124')


def _load_revo3_sysid(profile):
    root = DATA_DIR / profile
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, digest in provenance['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Identified hand checksum mismatch ({profile}): {name}')
    raw = json.loads((root / 'isaaclab_actuator_params.json').read_text())
    if raw.get('format') != 'revo3-sysid-isaaclab-v1' or raw.get('side') != 'right':
        raise ValueError(f'Unsupported revo3-sysid format in {profile}')

    def ordered(field):
        table = raw[field]
        if set(table) != set(REVO3_JOINT_NAMES):
            raise ValueError(f'revo3-sysid field {field} must cover all 21 Revo3 joints')
        vals = [float(table[n]) for n in REVO3_JOINT_NAMES]
        if not all(math.isfinite(x) for x in vals) or min(vals) < 0:
            raise ValueError(f'Invalid revo3-sysid values in {field}')
        return vals

    delay_s = float(raw['delay_summary']['mean_delay_s'])
    dt = 0.005
    delay = math.floor(delay_s / dt + 0.5)
    if not 0 <= delay < 10:
        raise ValueError('revo3-sysid mean delay must be shorter than 50 ms')
    p = dict(
        armature_kg_m2=ordered('armature'),
        viscous_Nm_s_rad=ordered('viscous_friction'),
        friction_Nm=ordered('friction'),
        encoder_bias_rad=[0.0] * len(REVO3_JOINT_NAMES),
        delay_s=delay_s,
        effective_kp_Nm_rad=ordered('stiffness'),
        effective_kd_Nm_s_rad=ordered('damping'),
        effective_effort_limit_Nm=ordered('effort_limit_peak'),
        effort_limit_cont_Nm=ordered('effort_limit_cont'),
        finger_body_inertia_scale=1.0,
    )
    info = dict(
        provenance,
        pd_period_s=dt,
        physics_substeps=2,
        delay_steps=delay,
        effective_delay_s=delay * dt,
        reference_mit_gains=raw.get('reference_mit_gains'),
        thumb_differential=raw.get('thumb_differential'),
        passive_spring_stiffness=raw.get('passive_spring_stiffness'),
    )
    return p, info


def load_identified_hand(profile='fit_03'):
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f'Unknown identified_hand_profile: {profile!r}; expected one of {SUPPORTED_PROFILES}')
    if profile != 'fit_03':
        return _load_revo3_sysid(profile)
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, digest in provenance['sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Identified hand checksum mismatch: {name}')
    p = json.loads((ROOT / 'best_parameters.json').read_text())
    h = json.loads((ROOT / 'hand.json').read_text())
    frozen = json.loads((ROOT / 'selection_frozen.json').read_text())
    if frozen['parameters_sha256'] != provenance['sha256']['best_parameters.json']:
        raise ValueError('Frozen model selection does not match parameter bytes')
    if h['joint_names'] != list(REVO3_JOINT_NAMES):
        raise ValueError('PACE joint order differs from policy order')
    for key, values in p.items():
        if isinstance(values, list):
            if (len(values) != 21 or not all(math.isfinite(x) for x in values)
                    or (key != 'encoder_bias_rad' and min(values) < 0)):
                raise ValueError(f'Invalid identified field {key}')
    if p['finger_body_inertia_scale'] != 1.0:
        raise ValueError('Non-unit body inertia fit needs an explicit asset mapping')
    dt = h['dt_s']
    delay = math.floor(p['delay_s'] / dt + .5)
    if dt != .005 or h.get('physics_substeps', 2) != 2 or not 0 <= delay < 10:
        raise ValueError('The one-last-action model requires fixed delay shorter than 50 ms')
    return p, dict(provenance, pd_period_s=dt, physics_substeps=2,
                   delay_steps=delay, effective_delay_s=delay * dt)


def actuator_config(profile='fit_03', *, use_identified_effort_limits=False):
    from isaaclab.actuators import DelayedPDActuatorCfg
    from .output_limits import load_output_limits
    p, info = load_identified_hand(profile)
    def joints(key):
        return dict(zip(REVO3_JOINT_NAMES, p[key]))
    if use_identified_effort_limits or profile != 'fit_03':
        effort = joints('effective_effort_limit_Nm')
        return DelayedPDActuatorCfg(
            joint_names_expr=list(REVO3_JOINT_NAMES),
            stiffness=joints('effective_kp_Nm_rad'), damping=joints('effective_kd_Nm_s_rad'),
            armature=joints('armature_kg_m2'), friction=joints('friction_Nm'),
            viscous_friction=joints('viscous_Nm_s_rad'),
            actuator_effort_limit=effort,
            joint_effort_limit=effort,
            actuator_velocity_limit=None,
            joint_velocity_limit=None,
            min_delay=info['delay_steps'], max_delay=info['delay_steps'])
    limits = load_output_limits()
    def uniform(value):
        # Keep per-name dictionaries so historical YAML profiles can still be
        # merged through Isaac Lab's typed configuration loader.
        return dict.fromkeys(REVO3_JOINT_NAMES, value)
    return DelayedPDActuatorCfg(
        joint_names_expr=list(REVO3_JOINT_NAMES),
        stiffness=joints('effective_kp_Nm_rad'), damping=joints('effective_kd_Nm_s_rad'),
        armature=joints('armature_kg_m2'), friction=joints('friction_Nm'),
        viscous_friction=joints('viscous_Nm_s_rad'),
        actuator_effort_limit=uniform(limits['effort_limit_Nm']),
        joint_effort_limit=uniform(limits['effort_limit_Nm']),
        actuator_velocity_limit=uniform(limits['velocity_limit_rad_s']),
        joint_velocity_limit=uniform(limits['velocity_limit_rad_s']),
        min_delay=info['delay_steps'], max_delay=info['delay_steps'])
