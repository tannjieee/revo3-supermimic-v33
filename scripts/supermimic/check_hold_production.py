"""Full-size training preflight with read-only per-control-step capacity checks.

Uses train.py and all production batch sizes. --check_joint_update exercises
Actor/Critic immediately for diagnostics; these weights are never promoted.
"""
import runpy
import json
from pathlib import Path
import numpy as np
import torch

ns = runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='hold_production_check')
from isaaclab_newton.physics import NewtonManager
from viserdex.supermimic.simulation import Adapter

args = ns['args']
if args.smoke or not args.check_joint_update or args.num_envs != 2048 or ns['world_size'] != 2:
    raise ValueError('Preflight requires two ranks, 2048 environments/rank and --check_joint_update, without --smoke')
original_step = Adapter.step
report = dict(passed=False,steps=0,num_envs_per_rank=args.num_envs,rank=ns['rank'],
              world_size=ns['world_size'],overflow_bits=0,
              peaks=dict(candidates=0,contacts=0,constraints=0))

def audited_step(self, action):
    if report['steps'] == 0:
        assert self.env.cfg.rewards.hand_pose.weight == -.2
        assert self.hold_cfg.hand_pose_weight == .2
        report['hand_pose_reward_weight'] = -.2
        assert self.env.cfg.scene.robot.actuators['fingers'].stiffness == 3.0
        lo, hi = self.context_schema['actuator.fingers.kp']
        kp = self.context[:, lo:hi]
        assert torch.isfinite(kp).all()
        assert float(kp.min()) >= 1.8 - 1e-5 and float(kp.max()) <= 4.5 + 1e-5
        report['actual_kp_range'] = [float(kp.min()), float(kp.max())]
        report['nominal_kp'] = 3.0
    row = original_step(self, action)
    data = NewtonManager._solver.mjw_data
    overflow = int(np.bitwise_or.reduce(data.overflow.numpy().ravel()))
    report['overflow_bits'] |= overflow
    assert overflow == 0, f'MJWarp capacity overflow: {overflow}'
    for name, value in (('candidates',data.ncollision),('contacts',data.nacon),('constraints',data.nefc)):
        report['peaks'][name] = max(report['peaks'][name], int(value.numpy().max()))
    assert torch.isfinite(row[0]['next_physical']).all()
    assert torch.isfinite(row[0]['reward']).all()
    report['peak_hand_pose_cost'] = max(report.get('peak_hand_pose_cost', 0.), float(self.captured_hand_pose.max()))
    report['hand_pose_parity_error'] = self.max_errors['hand_pose']
    report['steps'] += 1
    return row

Adapter.step = audited_step
try:
    ns['main']()
    report['passed'] = True
finally:
    rank_dir = args.output_dir / (f"rank_{ns['rank']}" if ns['rank'] else '')
    rank_dir.mkdir(parents=True,exist_ok=True)
    (rank_dir/'capacity_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HOLD_PRODUCTION_PREFLIGHT',json.dumps(report),flush=True)
