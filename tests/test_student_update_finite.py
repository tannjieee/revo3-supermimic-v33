"""The performance path still rejects non-finite losses before optimizer steps."""

import pytest
import torch

from viserdex.distillation.config import DistillationCfg
from viserdex.distillation.trainer import DAggerTrainer
from test_distillation import make_batch, small_model


def test_nonfinite_chunk_cannot_advance_optimizer_or_parameters():
    torch.manual_seed(22)
    model = small_model()
    trainer = DAggerTrainer(model, DistillationCfg(epochs=2, update_env_batch_size=2))
    batch = make_batch(model)
    batch["teacher"][:, -1].fill_(float("nan"))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(FloatingPointError):
        trainer.update(batch)
    assert trainer.iteration == 0 and trainer.total_steps == 0
    assert not trainer.optimizer.state
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
