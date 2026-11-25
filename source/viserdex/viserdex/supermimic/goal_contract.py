"""Resolve the serialized position command without changing saved environments."""
import numpy as np


def goal_position_metadata(cfg):
    """Canonical hand-base meters; distinct from the network's fixed origin."""
    explicit = cfg.commands.object_pose.goal_position_base_m
    if explicit is None:
        from scipy.spatial.transform import Rotation
        hand, obj = cfg.scene.robot.init_state, cfg.scene.object.init_state
        # ReorientCommand adds this offset in environment/world axes before
        # SingleGoalCommand optionally replaces it with an explicit base pose.
        offset = np.asarray(getattr(cfg.commands.object_pose, 'init_pos_offset', (0., 0., 0.)), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('init_pos_offset must contain three finite meters')
        position = Rotation.from_quat(hand.rot).inv().apply(np.asarray(obj.pos)+offset-np.asarray(hand.pos))
        source = 'object_default_root_pose_plus_command_offset' if offset.any() else 'object_default_root_pose'
    else:
        position = np.asarray(explicit, dtype=float)
        source = 'explicit_config'
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('goal_position_base_m must contain three finite meters')
    return dict(default_goal_position_base_m=[float(v) for v in position],
                default_goal_position_source=source)


def verify_saved_goal_position(cfg, requested):
    """An explicit CLI request may match, but cannot replace, a saved command."""
    requested = np.asarray(requested, dtype=float)
    if requested.shape != (3,) or not np.isfinite(requested).all():
        raise ValueError('goal_position_base_m must contain three finite meters')
    saved = goal_position_metadata(cfg)['default_goal_position_base_m']
    if not np.allclose(requested, saved, atol=1e-6, rtol=0):
        raise ValueError(f'Saved environment goal_position_base_m is {saved}; '
                         'cannot override its goal position on resume. Start a fresh run.')
