"""Learning-rate schedules independent of the simulator and Lightning."""

import math

from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(optimizer, warmup_steps=500, decay_steps=200, min_factor=0.2):
    """Warm up from 1% to full LR, decay to the floor, then keep it fixed."""
    if warmup_steps <= 0 or decay_steps <= 0 or not 0 <= min_factor <= 1:
        raise ValueError("Require positive schedule lengths and min_factor in [0, 1].")

    def factor(step):
        if step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        return min_factor + (1.0 - min_factor) * (1 + math.cos(math.pi * progress)) / 2

    return LambdaLR(optimizer, factor)
