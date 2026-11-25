"""Real-collection-only, bounded smooth random HOME targets in physical radians.

No changes to reset states, policy range, simulator PD, or imagined dynamics.
Runtime phase is reset with each environment, like fresh replay on resume.
"""
import torch


def smoothstep5(t):
    return t*t*t*(10+t*(-15+6*t))


class HomeExcitation:
    def __init__(self, limits, cfg):
        self.cfg = cfg
        self.lower, self.upper = limits.unbind(-1)
        self.half = (self.upper-self.lower)/2
        self.home = limits.new_tensor(cfg.action_home)-limits.new_tensor(cfg.encoder_bias_rad)
        margin = torch.minimum(self.home-self.lower, self.upper-self.home)
        if not torch.isfinite(limits).all() or not (margin > 0).all():
            raise ValueError('HOME excitation requires HOME strictly within finite encoder limits')
        # Bound the excitation itself, retaining the full production action range.
        self.amplitude = torch.minimum(torch.full_like(margin, cfg.warmup_joint_amplitude_rad), margin*.95)
        self.start = torch.zeros_like(margin)
        self.end = torch.zeros_like(margin)
        self.tick = torch.zeros(len(limits), dtype=torch.long, device=limits.device)

    @torch.no_grad()
    def action(self, iteration):
        if not 0 <= iteration < self.cfg.wm_only_iters:
            raise ValueError('HOME excitation is restricted to the warmup phase')
        beginning = self.tick == 0
        # Every environment has its own segment phase; reset starts at HOME.
        random_end = (2*torch.rand_like(self.end)-1)*self.amplitude
        self.end = torch.where(beginning[:, None], random_end, self.end)
        phase = self.tick.to(self.start.dtype)/self.cfg.warmup_joint_segment_steps
        offset = torch.lerp(self.start, self.end, smoothstep5(phase)[:, None])
        # Last 20% returns smoothly to HOME before the still-frozen Actor takes over.
        remaining = (self.cfg.wm_only_iters-1-iteration)/max(1., .2*(self.cfg.wm_only_iters-1))
        envelope = smoothstep5(min(1., max(0., remaining)))
        target = self.home+offset*envelope
        action = (target-(self.lower+self.upper)/2)/self.half
        self.tick += 1
        completed = self.tick > self.cfg.warmup_joint_segment_steps
        self.start = torch.where(completed[:, None], self.end, self.start)
        self.tick = torch.where(completed, 0, self.tick)
        return action

    @torch.no_grad()
    def reset(self, done):
        self.tick = torch.where(done, 0, self.tick)
        self.start = torch.where(done[:, None], 0., self.start)
        self.end = torch.where(done[:, None], 0., self.end)
