"""Frozen PACE fit, copied into this project and verified before use (SI units)."""
from pathlib import Path
import hashlib
import json
import math
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

ROOT = Path(__file__).parent / 'data' / 'fit_03'


def load_identified_hand():
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


def actuator_config():
    from isaaclab.actuators import DelayedPDActuatorCfg
    from .output_limits import load_output_limits
    p, info = load_identified_hand()
    limits = load_output_limits()
    def joints(key):
        return dict(zip(REVO3_JOINT_NAMES, p[key]))
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
