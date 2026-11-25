"""Batched sequence evaluation preserves the stepwise policy and BPTT resets."""

import copy

import pytest
import torch

from test_distillation import small_model


@pytest.mark.parametrize("reset_case", ["none", "sparse", "all", "first_and_last"])
def test_batched_sequence_matches_stepwise_outputs_hidden_and_gradients(reset_case):
    torch.manual_seed(43)
    torch.set_num_threads(1)
    batched = small_model().double()
    reference = copy.deepcopy(batched)
    prop = torch.randn(7, 3, 113, dtype=torch.float64)
    ext = torch.randn(7, 3, 11, dtype=torch.float64)
    initial = torch.randn(2, 3, 16, dtype=torch.float64)
    resets = torch.zeros(7, 3, dtype=torch.bool)
    if reset_case == "sparse":
        resets[2, 0] = True
        resets[4, 2] = True
    elif reset_case == "all":
        resets[:] = True
    elif reset_case == "first_and_last":
        resets[0] = True
        resets[-1, 1] = True
    hidden = initial.clone()
    expected_action, expected_recon = [], []
    for step in range(7):
        hidden = hidden * (~resets[step])[None, :, None]
        action, reconstruction, hidden = reference(prop[step], ext[step], hidden)
        expected_action.append(action)
        expected_recon.append(reconstruction)
    expected = torch.stack(expected_action), torch.stack(expected_recon), hidden
    actual = batched.sequence(prop, ext, initial, resets)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)
    sum(value.square().mean() for value in actual).backward()
    sum(value.square().mean() for value in expected).backward()
    for (_, left), (_, right) in zip(batched.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-9)
