"""Exercise the bound training task at full environment count; never optimize.

Reuse train.py's official launcher, saved environment binding, and task config.
The low-level MJWarp arrays below are read-only diagnostics. Solver capacity is
set exclusively through public env.sim.physics.solver_cfg CLI overrides.
"""
import json
from pathlib import Path
import runpy
import time

ns = runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='contact_stress')
args, app = ns['args'], ns['app']
import gymnasium as gym
import numpy as np
import torch
from isaaclab_newton.physics import NewtonManager

assert not args.resume and not args.evaluate
torch.manual_seed(args.seed + ns['rank'])
env = gym.make(ns['TASK_ID'], cfg=app.cfg).unwrapped
out = args.output_dir / f"stress_rank_{ns['rank']}.json"
out.parent.mkdir(parents=True, exist_ok=True)
result = dict(num_envs=env.num_envs, control_hz=1 / env.step_dt,
              binding=ns['binding'].verify(app.cfg), phases=[], passed=False)
try:
    env.reset()
    solver = NewtonManager._solver
    data = solver.mjw_data
    result.update(naconmax=int(data.naconmax), njmax=int(data.njmax))
    corners = torch.where(torch.rand(env.num_envs, 21, device=env.device) > .5, 1., -1.)
    action = torch.zeros_like(corners)
    with torch.inference_mode():
        for phase in ('zero', 'positive_limits', 'negative_limits', 'mixed_limits',
                      'reverse_mixed_limits', 'alternating_limits', 'random', 'positive_limits_again'):
            env.reset()
            peak_pairs, peak_contacts, peak_constraints, resets = 0, 0, 0, 0
            start = time.monotonic()
            for step in range(96):
                if phase == 'zero': action.zero_()
                elif phase.startswith('positive'): action.fill_(1.)
                elif phase == 'negative_limits': action.fill_(-1.)
                elif phase == 'mixed_limits': action.copy_(corners)
                elif phase == 'reverse_mixed_limits': action.copy_(-corners)
                elif phase == 'alternating_limits': action.fill_(1. if (step // 8) % 2 else -1.)
                else: action.uniform_(-1., 1.)
                obs, reward, terminated, timeout, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
                overflow = int(np.bitwise_or.reduce(data.overflow.numpy().ravel()))
                assert overflow == 0, (phase, step, overflow)
                peak_pairs = max(peak_pairs, int(data.ncollision.numpy().max()))
                peak_contacts = max(peak_contacts, int(data.nacon.numpy().max()))
                peak_constraints = max(peak_constraints, int(data.nefc.numpy().max()))
                resets += int((terminated | timeout).sum())
            row = dict(phase=phase, steps=96, peak_collision_candidates=peak_pairs,
                       peak_contacts=peak_contacts, peak_constraints_per_env=peak_constraints,
                       resets=resets, seconds=time.monotonic()-start)
            result['phases'].append(row)
            out.write_text(json.dumps(result, indent=2))
            print('CONTACT_STRESS_PHASE', json.dumps(row), flush=True)
    result['passed'] = True
    print('CONTACT_STRESS_PASSED', out, flush=True)
finally:
    out.write_text(json.dumps(result, indent=2))
    env.close()
    app.close()
