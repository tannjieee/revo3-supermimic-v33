"""Frozen, read-only SDK software position limits in policy encoder coordinates.

Software command limits are not the USD's geometric mechanical stops. The
identified encoder bias is applied only when converting a geometric pose.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

DEFAULT_LIMITS = Path(__file__).resolve().parents[4] / 'config/revo3_sdk_limits.json'
SCHEMA = 'revo3_sdk_position_limits_v1'


def validate_limits(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (21, 2) or not np.isfinite(value).all()
            or not (value[:, 1] > value[:, 0]).all()):
        raise ValueError('Expected 21 finite, ordered joint limit pairs')
    return value


def policy_limits(sdk_names, sdk_limits_deg, offsets_sdk_rad):
    if len(sdk_names) != 21 or set(sdk_names) != set(REVO3_JOINT_NAMES):
        raise ValueError('SDK joint names must match the 21 policy joints exactly')
    offsets = np.asarray(offsets_sdk_rad, dtype=float)
    if offsets.shape != (21,) or not np.isfinite(offsets).all():
        raise ValueError('Expected 21 finite SDK-order offsets')
    sdk_limits = validate_limits(sdk_limits_deg)
    order = [sdk_names.index(n) for n in REVO3_JOINT_NAMES]
    return (np.deg2rad(sdk_limits) - offsets[:, None])[order]


def load_sdk_limits(path=DEFAULT_LIMITS):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (data.get('schema') != SCHEMA or data.get('joint_names') != list(REVO3_JOINT_NAMES)
            or data.get('coordinate') != 'policy_encoder_rad'
            or data.get('layout') != {'layout_id': 'Revo3Ultra21', 'version': 1, 'joint_count': 21}
            or data.get('sdk_version') != '2.3.2'
            or data.get('configured_signs') != [1] * 21):
        raise ValueError('Unsupported SDK software-limit contract')
    expected = policy_limits(data['sdk_joint_names'], data['sdk_limits_deg'], data['offsets_sdk_rad'])
    limits = validate_limits(data['joint_limits_rad'])
    if not np.allclose(limits, expected, atol=1e-12, rtol=0):
        raise ValueError('SDK degree/order/offset conversion does not match stored limits')
    if not data.get('serial') or not data.get('readback_sha256') or not data.get('captured_at'):
        raise ValueError('SDK limits require device/readback provenance')
    return limits, dict(schema=SCHEMA, sha256=hashlib.sha256(raw).hexdigest(),
                       serial=data['serial'], sdk_version=data['sdk_version'],
                       firmware=data['firmware'], captured_at=data['captured_at'],
                       readback_sha256=data['readback_sha256'], coordinate=data['coordinate'])


def sampling_geometry_limits(encoder_limits, bias, mechanical_limits):
    """Intersect SDK encoder bounds shifted by bias with physical USD bounds."""
    encoder_limits, mechanical_limits = validate_limits(encoder_limits), validate_limits(mechanical_limits)
    bias = np.asarray(bias, dtype=float)
    if bias.shape != (21,) or not np.isfinite(bias).all():
        raise ValueError('Expected 21 finite geometric-minus-encoder biases')
    shifted = encoder_limits + bias[:, None]
    return validate_limits(np.column_stack((np.maximum(shifted[:, 0], mechanical_limits[:, 0]),
                                           np.minimum(shifted[:, 1], mechanical_limits[:, 1]))))
