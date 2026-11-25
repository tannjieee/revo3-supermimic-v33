"""Real Newton lifecycle diagnostic; no optimizer or trained policy is loaded."""
import runpy,json
from pathlib import Path
ns=runpy.run_path(str(Path(__file__).with_name('train.py')),run_name='hold_environment_check')
args,app=ns['args'],ns['app']
import gymnasium as gym
import torch,numpy as np
from isaaclab_newton.physics import NewtonManager
from viserdex.supermimic.simulation import Adapter,tensor
from viserdex.supermimic.core import Q,HOLD
from viserdex.distillation.config import PerceptionNoiseCfg
args.output_dir.mkdir(parents=True,exist_ok=False)
env=gym.make(ns['TASK_ID'],cfg=app.cfg).unwrapped
result=dict(passed=False,network_updates=0,num_envs=env.num_envs,seed=args.seed,
    diagnostic='Fixed joint targets; align goal once after settling, then exercise real hold and timeout lifecycle')
try:
    torch.manual_seed(args.seed)
    clean,_=env.reset();a=Adapter(env,PerceptionNoiseCfg());a.audit_observation(clean)
    lo,hi=a.limits.unbind(-1)
    def hold_action():return torch.zeros_like(a.state[:,Q])
    action=hold_action();command=env.command_manager.get_term('object_pose')
    data=NewtonManager._solver.mjw_data
    total_events=0;max_ticks=0;partial_hold_reset=False;peaks=dict(candidates=0,contacts=0,constraints=0)
    with torch.no_grad():
        for step in range(150):
            if step==50:
                # Diagnostic command assignment only: physics/randomization are unchanged.
                command.quat_command_w.copy_(tensor(a.obj.data.root_quat_w))
                command.hold_ticks.zero_();a.state=a.read_state()
            if step==55:
                ids=(command.hold_ticks>0).nonzero().flatten()[::2]
                if len(ids):
                    partial_hold_reset=True
                    env.episode_length_buf[ids]=env.max_episode_length-1
                    a.state=a.read_state()
            row,_,_,clean=a.step(action)
            max_ticks=max(max_ticks,float(a.state[:,HOLD].max()))
            total_events+=int(row['success'].sum())
            assert torch.isfinite(a.state).all() and torch.isfinite(row['reward']).all()
            assert int(np.bitwise_or.reduce(data.overflow.numpy().ravel()))==0
            peaks['candidates']=max(peaks['candidates'],int(data.ncollision.numpy().max()))
            peaks['contacts']=max(peaks['contacts'],int(data.nacon.numpy().max()))
            peaks['constraints']=max(peaks['constraints'],int(data.nefc.numpy().max()))
            if row['done'].any():action[row['done']]=hold_action()[row['done']]
    assert total_events>0, 'Need an actual successful hold, not just zero-event parity'
    assert max_ticks==9 and partial_hold_reset
    env.episode_length_buf[:]=env.max_episode_length-1;a.state=a.read_state()
    row,_,_,clean=a.step(action)
    assert row['done'].all() and not row['success'].any() and not a.state[:,HOLD].any()
    a.audit_observation(clean)
    min_pose_reward = 0.0
    with torch.no_grad():
        for probe_step in range(24):
            action = torch.full_like(action, .7 if probe_step % 8 < 4 else -.7)
            if probe_step == 12:
                env.episode_length_buf[::2] = env.max_episode_length - 1
                a.state = a.read_state()
            row,_,_,clean = a.step(action)
            assert int(np.bitwise_or.reduce(data.overflow.numpy().ravel())) == 0
            min_pose_reward = min(min_pose_reward, -a.hold_cfg.hand_pose_weight*env.step_dt*float(a.captured_hand_pose.max()))
    assert min_pose_reward < -1e-6
    result['hand_pose_probe_min_reward'] = min_pose_reward
    result['hand_pose_weight'] = a.hold_cfg.hand_pose_weight
    result.update(passed=True,total_hold_completions=total_events,max_partial_hold_ticks=max_ticks,
        partial_hold_reset_verified=partial_hold_reset,forced_all_timeout_verified=True,
        state_dim=a.state.shape[-1],actor_history_shape=list(a.history.shape[1:]),
        max_errors=a.max_errors,contact_peaks=peaks,live_convex_audit=a.convex_audit)
    print('HOLD_ENVIRONMENT_PASSED',json.dumps(result),flush=True)
finally:
    (args.output_dir/'validation.json').write_text(json.dumps(result,indent=2))
    env.close();app.close()
