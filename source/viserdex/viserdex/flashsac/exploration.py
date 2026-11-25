"""S0 keeps the full absolute action map. Reject jumps before the physics step."""
import torch
from viserdex.supermimic.state_layout import COMPACT


class ActionRejected(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f'Absolute command rejected before env.step: {report}')


def audit_action(action, state, limits, cfg):
    if not torch.isfinite(action).all() or (action.abs() > 1).any():
        raise ActionRejected(dict(reason='nonfinite_or_out_of_range'))
    lo, hi = limits.unbind(-1)
    target = lo+(action+1)*.5*(hi-lo)
    previous = lo+(state[:, COMPACT.last_action]+1)*.5*(hi-lo)
    previous = torch.where(state[:, COMPACT.delay_ready].bool(), previous,
                           state[:, :21]-state.new_tensor(cfg.encoder_bias_rad))
    delta = (target-previous).abs()
    allowed = cfg.cost_target_rate_limit_rad_s*cfg.control_dt
    value, index = delta.flatten().max(0)
    report = dict(max_delta_deg=float(value)*180/torch.pi,
                  limit_deg=allowed*180/torch.pi, env_id=int(index)//21, joint_id=int(index)%21)
    if float(value) > allowed+2e-6:
        raise ActionRejected(dict(reason='target_rate', **report))
    return report
