"""The migrated schedule must stay finite and never restart after decay."""

import pytest
import torch

from viserdex.utils.schedules import warmup_cosine


def test_warmup_decay_and_floor():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    scheduler = warmup_cosine(optimizer)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(1200):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(1e-6)
    assert rates[500] == pytest.approx(1e-4)
    assert rates[700] == pytest.approx(2e-5)
    assert all(a <= b for a, b in zip(rates[:500], rates[1:501]))
    assert all(a >= b for a, b in zip(rates[500:], rates[501:]))
    assert rates[700:] == pytest.approx([2e-5] * 501)
