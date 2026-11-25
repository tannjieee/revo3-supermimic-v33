"""Paired open-loop Newton response probe; normal launcher and all guards retained.

Environment: RESPONSE_OUTPUT, RESPONSE_SIGN (-1/0/1), RESPONSE_REFERENCE
(optional baseline .pt), RESPONSE_EPS_DEG (default .5). Only control step 128
is perturbed. All later commands are replayed from baseline, not feedback.
Each world tests one joint; randomization is untouched. Compare prefixes before
interpreting finite differences; this is a directional response, not full J.
"""
import os, sys
from pathlib import Path
import runpy
import torch
from viserdex.supermimic.simulation import Adapter

if '--evaluate' not in sys.argv or '--resume' not in sys.argv:
    raise ValueError('Read-only diagnostic requires --evaluate and --resume')
out = Path(os.environ['RESPONSE_OUTPUT'])
sign = int(os.environ.get('RESPONSE_SIGN', '0'))
eps = float(os.environ.get('RESPONSE_EPS_DEG', '.5'))
reference = os.environ.get('RESPONSE_REFERENCE')
reference = torch.load(reference, weights_only=False) if reference else None
original = Adapter.step
records = []

def step(self, action):
    index = len(records)
    if reference is not None:
        action = reference['records'][index]['action'].to(action.device).clone()
    if index == 128 and sign:
        joint = torch.arange(len(action), device=action.device) % 21
        row = torch.arange(len(action), device=action.device)
        half = (self.limits[row, joint, 1]-self.limits[row, joint, 0])/2
        delta = eps * torch.pi/180 / half
        eligible = action[row, joint].abs()+delta < 1
        action[row, joint] += sign * delta * eligible
        if (action.abs() > 1).any():
            raise RuntimeError('Perturbation exceeds original action range')
    result = original(self, action)
    record = result[0]
    records.append({k: v.detach().cpu().clone() for k,v in record.items()})
    if index == 135:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(records=records, context=self.context.cpu(), limits=self.limits.cpu(),
                        sign=sign, eps_deg=eps, probe_step=128), out)
    return result

Adapter.step = step
runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
