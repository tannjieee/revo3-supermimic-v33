"""Training-only multi-step sensitivity check, separate from fit/model selection.

This uses a fixed weighted SI readout, not a task reward or a real-gradient label.
Finite differences across several radii expose local curvature/branch changes.
"""
import argparse
import json
from pathlib import Path

import torch

from experiment_contact_sdf import build
from experiment_contact_features import Config, COMPACT, action_target, advance, physical_delta, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads((args.experiment / 'plan.json').read_text())
    payload = torch.load(plan['checkpoint'], weights_only=False, map_location='cpu')
    surface = json.loads((args.experiment / 'surface.json').read_text())
    cfg = Config(**payload['config'])
    data = torch.load(args.experiment / 'branches_seed60/32/base.pt', weights_only=False, map_location='cpu')
    state = data['records'][0]['state'][:2].double()
    context, limits = data['context'][:2].double(), data['limits'][:2].double()
    direction = torch.linspace(-.5, .5, 42, dtype=torch.double).reshape(2, 21)
    weights = torch.linspace(.3, 1., 54, dtype=torch.double)
    report = {'scope': 'Initial models; first two training states; perturb first target only; not real physics gradient accuracy', 'models': {}}
    for arm in ('quasi', 'dynamic', 'baseline'):
        model = build(arm, payload, surface, 'cpu').double()

        def evaluate(delta, horizon):
            current = state
            for t in range(horizon):
                action = data['records'][t]['action'][:2].double()
                target = action_target(current, action, limits, cfg)
                if t == 0:
                    target = target + delta * direction
                    action = (target - limits.mean(-1)) / ((limits[..., 1] - limits[..., 0]) / 2)
                prediction = model(current, target, context)
                current = advance(current, prediction, action, target, current[:, COMPACT.goal_rot], 500, hold_cfg=cfg)
            return (physical_delta(state, prediction) * weights).sum(), prediction

        rows = []
        for horizon in (1, 2, 4, 8):
            delta = torch.tensor(0., dtype=torch.double, requires_grad=True)
            value, prediction = evaluate(delta, horizon)
            gradients = torch.autograd.grad(value, (delta, *model.parameters()), allow_unused=True)
            directional = float(gradients[0])
            parameter_norm = sum(float(g.detach().square().sum()) for g in gradients[1:] if g is not None) ** .5
            finite = {}
            with torch.no_grad():
                for epsilon in (1e-4, 1e-5, 1e-6, 1e-7, 1e-9, 1e-11, 1e-13):
                    derivative = float((evaluate(epsilon, horizon)[0] - evaluate(-epsilon, horizon)[0]) / (2 * epsilon))
                    finite[str(epsilon)] = {'derivative': derivative, 'relative_disagreement': abs(directional - derivative) / max(1., abs(directional), abs(derivative))}
            row = {'horizon': horizon, 'readout': float(value.detach()), 'first_target_directional_derivative': directional,
                   'parameter_gradient_norm': parameter_norm, 'finite_difference': finite,
                   'max_object_angular_speed_rad_s': float(prediction[:, 52:55].detach().norm(dim=-1).max())}
            rows.append(row)
            print(arm, json.dumps(row), flush=True)
        report['models'][arm] = rows
    save_json(args.experiment / 'rollout_gradient_check.json', report)


if __name__ == '__main__':
    main()
