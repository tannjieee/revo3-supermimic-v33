"""User-defined training bounds, independently stored and bound to SDK/PACE data."""
import hashlib
import json
from pathlib import Path
import numpy as np
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from .sdk_limits import DEFAULT_LIMITS, load_sdk_limits, validate_limits
from .identified_hand import load_identified_hand

DEFAULT_TRAINING_LIMITS = DEFAULT_LIMITS.with_name('revo3_training_limits.json')


def load_training_limits(path=DEFAULT_TRAINING_LIMITS, sdk_path=DEFAULT_LIMITS):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (data.get('schema') != 'revo3_training_position_limits_v1'
            or data.get('coordinate') != 'policy_encoder_rad'
            or data.get('authoritative_units') != 'deg'
            or data.get('joint_names') != list(REVO3_JOINT_NAMES)):
        raise ValueError('Unsupported training software-limit contract')
    degrees = validate_limits(data['joint_limits_deg'])
    limits = validate_limits(data['joint_limits_rad'])
    if not np.allclose(limits, np.deg2rad(degrees), atol=1e-12, rtol=0):
        raise ValueError('Training degrees and radians disagree')
    sdk, sdk_source = load_sdk_limits(sdk_path)
    if (data.get('sdk_limits_sha256') != sdk_source['sha256']
            or data.get('serial') != sdk_source['serial']):
        raise ValueError('Training limits refer to a different SDK/device snapshot')
    if ((limits[:, 0] < sdk[:, 0] - 1e-12) | (limits[:, 1] > sdk[:, 1] + 1e-12)).any():
        raise ValueError('Training limits must be contained in SDK software limits')
    _, identified = load_identified_hand()
    if data.get('identified_parameters') != dict(
            model=identified['model'], sha256=identified['sha256']['best_parameters.json']):
        raise ValueError('Training limits refer to different identified parameters')
    return limits, dict(schema=data['schema'], sha256=hashlib.sha256(raw).hexdigest(),
                        coordinate=data['coordinate'], defined_by=data['defined_by'],
                        defined_on=data['defined_on'], sdk_software_limits=sdk_source,
                        identified_parameters=data['identified_parameters'])
