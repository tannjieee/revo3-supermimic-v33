"""Explicit transition versus displayed-frame semantics for recorded rollouts."""

from .state_layout import layout


def trajectory_contract():
    """Describe every NPZ field; retain the historical field names."""
    return dict(
        protocol='transition_and_displayed_frame_v1',
        quaternion_order='XYZW',
        selection='one sample per displayed frame, environment zero',
        physical_layout=dict(joint_position_geometry_rad=[0, 21], joint_velocity_rad_s=[21, 42],
            object_position_base_m=[42, 45], object_quaternion_xyzw_base=[45, 49],
            object_linear_velocity_base_m_s=[49, 52], object_angular_velocity_base_rad_s=[52, 55]),
        fields=dict(
            actions='normalized absolute command applied during the completed transition',
            raw_actions='Actor command before optional playback target shaping',
            physical='pre-reset next physical state in hand-base coordinates; historical field retained',
            target_quaternions_xyzw='displayed post-step/post-reset goal orientation in world coordinates',
            current_quaternions_xyzw='displayed post-step/post-reset object orientation in world coordinates',
            done='completed transition ended by termination or timeout',
            terminated='completed transition ended by task termination, excluding timeout',
            post_step_physical='displayed post-step/post-reset physical state in hand-base coordinates',
            transition_goal_quaternion_xyzw_base='goal used by the completed transition, before resampling/reset, in hand-base coordinates',
            base_quaternion_xyzw_world='hand-base orientation in world coordinates for this displayed frame'),
        reset_semantics='When done is true, physical is the terminal state and post_step_physical is the new reset state; displayed quaternions belong to post_step_physical.',
    )


def trajectory_snapshot(row, raw_action, post_step_state, current_quaternion_world,
                        target_quaternion_world, base_quaternion_world):
    """Copy one transition and its displayed state before simulator buffers change."""
    values = dict(actions=row['action'], raw_actions=raw_action, physical=row['next_physical'],
                  target_quaternions_xyzw=target_quaternion_world,
                  current_quaternions_xyzw=current_quaternion_world,
                  done=row['done'], terminated=row['terminated'],
                  post_step_physical=post_step_state[:, :55],
                  transition_goal_quaternion_xyzw_base=row['state'][:, layout(row['state']).goal_rot],
                  base_quaternion_xyzw_world=base_quaternion_world)
    # CPU tensors share storage with numpy, so cpu() alone is insufficient.
    return {name: value[0].detach().cpu().numpy().copy() for name, value in values.items()}
