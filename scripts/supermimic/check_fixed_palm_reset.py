"""Bounded Newton HOME-hold verification of fixed initial states and partial reset."""
import argparse
import json
from pathlib import Path
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.manual_convex import TASK_ID
from viserdex.supermimic.revised_contract import configure_environment, CENTER_REVISION
from viserdex.supermimic.simulation import TransitionRecorderCfg
from viserdex.supermimic.object_variant import bind_object
from viserdex.supermimic.fixed_palm_reset import configure_fixed_palm_reset, POSITION_BASE_M, QUATERNION_BASE_XYZW

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--object_variant', choices=('die50', 'die50_47g', 'die58'), default='die50_47g')
parser.add_argument('--steps', type=int, default=550)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--candidate_file', type=Path, help='Explicit offline pose experiment; never modifies the production profile')
parser.add_argument('--candidate_index')
add_runtime_args(parser, TASK_ID)
args = parse_runtime_args(parser)
if args.num_envs < 2 or args.steps < 550:
    parser.error('Use at least two environments and 550 steps to cover partial/natural timeout resets')
if (args.candidate_file is None) != (args.candidate_index is None):
    parser.error('Candidate experiments require both a file and an explicit candidate index')
if args.candidate_file is not None:
    from viserdex.supermimic import fixed_palm_reset
    candidates = json.loads(args.candidate_file.read_text())
    selected = next(row for row in candidates if str(row['candidate_index']) == args.candidate_index)
    if not selected['success'] or selected['min_gap'] < 0:
        parser.error('Refusing a failed or penetrating offline candidate')
    POSITION_BASE_M = fixed_palm_reset.POSITION_BASE_M = tuple(float(v) for v in selected['p'])
    QUATERNION_BASE_XYZW = fixed_palm_reset.QUATERNION_BASE_XYZW = tuple(float(v) for v in selected['quat'])


def configure(cfg):
    configure_environment(cfg, CENTER_REVISION)
    bind_object(cfg, args.object_variant)
    configure_fixed_palm_reset(cfg)
    configure_clean_state_observations(cfg)
    cfg.recorders = TransitionRecorderCfg()
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = args.seed


app = SimulationRuntime(args, configure=configure)
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.io import dump_yaml
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import hand_center_config, quat_conjugate, quat_mul, rotation_log
from viserdex.supermimic.simulation import Adapter


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
    try:
        env.reset()
        cfg = hand_center_config()
        adapter = Adapter(env, PerceptionNoiseCfg(), policy_cfg=cfg)
        initial = adapter.state.clone()
        context = adapter.context.clone()
        from viserdex.supermimic.fixed_palm_reset import configured_pose_base
        position, rotation = configured_pose_base(env.cfg)
        local_position = initial.new_tensor(position)
        local_rotation = initial.new_tensor(rotation)
        home = initial.new_tensor(cfg.action_home)
        observed_resets = 0
        reset_position_error = 0.

        def check_reset(ids):
            nonlocal observed_resets, reset_position_error
            state = adapter.state[ids]
            torch.testing.assert_close(state[:, :21], home.expand(len(ids), -1), atol=1e-7, rtol=0)
            torch.testing.assert_close(state[:, 21:42], torch.zeros_like(state[:,21:42]), atol=0, rtol=0)
            torch.testing.assert_close(state[:,42:45], local_position.expand(len(ids), -1), atol=6e-6, rtol=0)
            angle = rotation_log(quat_mul(state[:,45:49], quat_conjugate(local_rotation.expand(len(ids), -1)))).norm(dim=-1)
            assert float(angle.max()) < 1e-6
            torch.testing.assert_close(state[:,49:76], torch.zeros_like(state[:,49:76]), atol=0, rtol=0)
            assert not state[:,83].any()
            reset_position_error = max(reset_position_error, float((state[:,42:45]-local_position).abs().max()))
            observed_resets += len(ids)

        check_reset(torch.arange(env.num_envs, device=env.device))
        goals = initial[:,79:83]
        assert len(torch.unique(goals, dim=0)) == env.num_envs
        lo, hi = adapter.limits.unbind(-1)
        action = 2*(home-initial.new_tensor(cfg.encoder_bias_rad)-lo)/(hi-lo)-1
        history, forced = [], None
        drops = timeouts = 0
        max_position_drift = max_rotation_drift = max_linear_speed = max_angular_speed = 0.
        displacement_low, displacement_high = torch.zeros(3,device=env.device), torch.zeros(3,device=env.device)
        with torch.no_grad():
            for step in range(args.steps):
                if step == 25:
                    env.episode_length_buf[:env.num_envs//2] = env.max_episode_length-1
                    old_goals = adapter.state[:,79:83].clone()
                row, _, _, _ = adapter.step(action)
                next_state = row['next_physical']
                displacement = next_state[:,42:45]-local_position
                displacement_low = torch.minimum(displacement_low, displacement.amin(0))
                displacement_high = torch.maximum(displacement_high, displacement.amax(0))
                position_drift = (next_state[:,42:45]-local_position).norm(dim=-1)
                rotation_drift = rotation_log(quat_mul(next_state[:,45:49],
                    quat_conjugate(local_rotation.expand(env.num_envs,-1)))).norm(dim=-1)
                max_position_drift = max(max_position_drift, float(position_drift.max()))
                max_rotation_drift = max(max_rotation_drift, float(rotation_drift.max()))
                max_linear_speed = max(max_linear_speed, float(next_state[:,49:52].norm(dim=-1).max()))
                max_angular_speed = max(max_angular_speed, float(next_state[:,52:55].norm(dim=-1).max()))
                drops += int(row['terminated'].sum())
                timeouts += int((row['done'] & ~row['terminated']).sum())
                if row['done'].any():
                    check_reset(row['done'].nonzero().flatten())
                if step == 25:
                    expected = torch.arange(env.num_envs, device=env.device) < env.num_envs//2
                    torch.testing.assert_close(row['done'], expected)
                    # No physics step occurs between the recorder's pre-reset
                    # capture and Adapter's post-reset state. Unselected worlds
                    # must retain every physical field exactly, not only goals.
                    torch.testing.assert_close(adapter.state[~expected, :55],
                                               row['next_physical'][~expected], atol=0, rtol=0)
                    torch.testing.assert_close(adapter.state[~expected,79:83], old_goals[~expected], atol=0, rtol=0)
                    assert (adapter.state[expected,79:83]-old_goals[expected]).abs().amax(-1).min() > 1e-5
                    forced = dict(step=step+1, reset_worlds=int(expected.sum()),
                                  selected_goals_changed=True, other_goals_unchanged=True,
                                  other_physical_states_unchanged=True)
                if step < 25 or (step+1) % 25 == 0:
                    history.append(dict(step=step+1, max_position_drift_m=float(position_drift.max()),
                        mean_displacement_base_m=displacement.mean(0).cpu().tolist(),
                        max_orientation_drift_deg=float(rotation_drift.max())*180/np.pi))
        live_context, schema = adapter.read_context()
        assert schema == adapter.context_schema
        torch.testing.assert_close(live_context, context, atol=0, rtol=0)
        ranges = {}
        for key in ('object.body_mass','object.shape_material_mu','actuator.fingers.kp','actuator.fingers.kd'):
            start, stop = schema[key]
            values = context[:,start:stop]
            ranges[key] = [float(values.min()), float(values.max())]
            assert float(values.std()) > 0
        # A finite holding check, not a perturbation/grasp certification.
        passed = bool(drops == 0 and max_position_drift < .005 and max_rotation_drift < np.deg2rad(5))
        report = dict(passed=passed, seed=args.seed, num_envs=env.num_envs, steps=args.steps,
            candidate_index=args.candidate_index, experimental_pose=args.candidate_file is not None,
            seconds_per_world=args.steps*env.step_dt, home_command_only=True, zero_action_exploration=True,
            initial_goal_unique_count=len(torch.unique(goals,dim=0)), observed_reset_worlds=observed_resets,
            drops=drops, timeouts=timeouts, forced_partial_reset=forced,
            max_reset_local_position_error_m=reset_position_error,
            max_position_drift_m=max_position_drift, max_orientation_drift_deg=max_rotation_drift*180/np.pi,
            displacement_bounds_base_m=[displacement_low.cpu().tolist(),displacement_high.cpu().tolist()],
            max_linear_speed_m_s=max_linear_speed, max_angular_speed_rad_s=max_angular_speed,
            thresholds=dict(max_position_drift_m=.005,max_orientation_drift_deg=5.,drops=0),
            domain_parameter_ranges=ranges, live_domain_parameters_unchanged=True,
            support_reset=env.object_support_audit,
            solver_diagnostic=adapter._solver_diagnostic(), parity_errors=adapter.max_errors,
            history=history, scope='finite HOME holding with physical domain randomization; no grasp library or reset settling')
        dump_yaml(str(args.output.with_suffix('.env.yaml')), env.cfg)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:value for key,value in report.items()
                          if key not in ('support_reset','history','solver_diagnostic','parity_errors')},indent=2))
        assert passed, 'Fixed palm HOME hold failed; see preserved report'
    finally:
        env.close()
        app.close()


if __name__ == '__main__':
    main()
