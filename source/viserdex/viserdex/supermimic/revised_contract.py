"""Version-26 little-first policy contract; vendor and legacy contracts stay immutable."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

REVISION = 'urdf_little_v2'
CENTER_REVISION = 'hardware_center25_v3'
JOINT_NAMES = tuple(f'right_{finger}_{joint}_joint'
    for finger in ('little', 'ring', 'middle', 'index')
    for joint in ('MPR', 'MCP', 'PIP', 'DIP')) + tuple(REVO3_JOINT_NAMES[:5])
HARDWARE_JOINT_NAMES = JOINT_NAMES[:16] + tuple(f'right_thumb_{j}_joint' for j in ('MCP','PIP','DIP','CMP','CMR'))
ORDER = tuple(REVO3_JOINT_NAMES.index(name) for name in JOINT_NAMES)
URDF = Path(__file__).resolve().parents[4] / 'config/revo3_right_limits_reference.urdf'


def joint_names(cfg):
    if cfg.network_revision == CENTER_REVISION:
        return HARDWARE_JOINT_NAMES
    return JOINT_NAMES if cfg.network_revision == REVISION else REVO3_JOINT_NAMES


def reorder(values):
    return [values[i] for i in ORDER]


def urdf_limits(revision=REVISION):
    from .identified_hand import load_identified_hand
    from .sdk_limits import load_sdk_limits
    names = HARDWARE_JOINT_NAMES if revision == CENTER_REVISION else JOINT_NAMES
    order = [REVO3_JOINT_NAMES.index(n) for n in names]
    raw = URDF.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '4a911b596499290aea7496a1a080ebfaa78f368f565e2e5f9c0475565a086289':
        raise ValueError('Pinned URDF reference changed; revise the policy contract explicitly')
    joints = {j.attrib['name']: j for j in ET.fromstring(raw).findall('joint') if j.get('type') == 'revolute'}
    if set(joints) != set(JOINT_NAMES):
        raise ValueError('URDF must contain exactly the 21 policy joints')
    geometric = np.array([[float(joints[n].find('limit').get(k)) for k in ('lower','upper')] for n in names])
    p, _ = load_identified_hand()
    bias = np.array([p['encoder_bias_rad'][i] for i in order])
    encoder = geometric - bias[:, None]
    if not np.isfinite(encoder).all() or not (encoder[:, 1] > encoder[:, 0]).all():
        raise ValueError('Invalid URDF position limits')
    _, sdk = load_sdk_limits()
    return encoder, dict(schema='revo3_urdf_encoder_limits_v2', sha256=hashlib.sha256(raw).hexdigest(),
        coordinate='policy_encoder_rad', source_urdf=str(URDF), source_revision='2a98621de70d59d182b8075214cebb96fd5375d3',
        joint_names=list(names), geometric_limits_rad=geometric.tolist(), encoder_bias_rad=bias.tolist(),
        conversion='encoder = URDF geometric angle - identified encoder bias',
        sdk_software_limits=sdk, sdk_containment_required=False)


def configure_environment(env_cfg, revision=REVISION):
    from .core import revised_manual_config
    from .core import hand_center_config
    cfg = hand_center_config() if revision == CENTER_REVISION else revised_manual_config()
    env_cfg.commands.object_pose.track_object_position = cfg.track_object_position
    env_cfg.commands.object_pose.stability_reward_scale = cfg.stability_reward_scale
    limits, source = urdf_limits(revision)
    action = env_cfg.actions.joint_pos
    action.limit_contract = revision
    action.joint_names = list(joint_names(cfg))
    action.preserve_order = True
    action.position_limits_rad = tuple(map(tuple, limits.tolist()))
    action.encoder_bias_rad = cfg.encoder_bias_rad
    action.training_limits_sha256 = source['sha256']
    action.software_limits_sha256 = source['sdk_software_limits']['sha256']

    if revision == CENTER_REVISION:
        from isaaclab.managers import TerminationTermCfg
        from .spatial_contract import object_away_from_hand_center
        env_cfg.sim.dt = .005
        env_cfg.decimation = 8
        env_cfg.sim.render_interval = 8
        env_cfg.commands.object_pose.hold_duration = cfg.hold_duration
        env_cfg.rewards.joint_vel_l2.weight = 0.
        env_cfg.rewards.action_rate_l2.weight = 0.
        env_cfg.terminations.object_out_of_reach = TerminationTermCfg(
            func=object_away_from_hand_center, params=dict(origin=cfg.hand_center_base_m, threshold=cfg.drop_radius_m))
