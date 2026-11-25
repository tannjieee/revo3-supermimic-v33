"""Replay storage changes execution placement, never sequence semantics.

CUDA checks are opt-in so ordinary CPU validation does not allocate on a live
training GPU: REPLAY_TEST_CUDA=1 pytest tests/test_replay_device.py.
"""
import os

import pytest
import torch

from viserdex.supermimic.replay import Replay


CUDA = pytest.mark.skipif(os.environ.get("REPLAY_TEST_CUDA") != "1",
                          reason="set REPLAY_TEST_CUDA=1 for scheduled GPU validation")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def transition(t, done=False, device="cpu"):
    state = torch.zeros(2, 84, device=device)
    state[:, 0] = t
    state[:, 48] = state[:, 82] = 1
    nxt = state[:, :55].clone()
    nxt[:, 42] = .25  # Event archive motion proxy; not a contact label.
    return dict(state=state, next_physical=nxt, next_aux=state[:, 55:].clone(),
                action=torch.full((2, 21), float(t), device=device),
                history=state[:, None, :8].expand(-1, 10, -1).clone(),
                **{"sensor/offset": torch.full((2, 3), float(t), device=device)},
                reward=torch.full((2,), float(t), device=device),
                terminated=torch.full((2,), done, device=device),
                done=torch.full((2,), done, device=device))


@pytest.mark.parametrize("adaptive", [False, True])
def test_explicit_cpu_keeps_default_sampling_rng(adaptive):
    default = Replay(6, 2, "cpu", event_capacity=0, event_window=3, adaptive=adaptive)
    explicit = Replay(6, 2, "cpu", event_capacity=0, event_window=3,
                      adaptive=adaptive, storage_device="cpu")
    for t in range(8):
        default.add(transition(t, t == 4))
        explicit.add(transition(t, t == 4))
    for window in (1, 3):
        torch.manual_seed(761)
        a = default.sample(127, window)
        torch.manual_seed(761)
        b = explicit.sample(127, window)
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert default.allocated_bytes == explicit.allocated_bytes
    assert default.storage_device == torch.device("cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_resident_ring_masks_fields_probabilities_and_real_sample(device):
    replay = Replay(6, 2, device, event_window=3, adaptive=True, storage_device=device)
    for t in range(8):
        replay.add(transition(t, t == 4, device))
    replay.priority[2] = 1000
    batch = replay.sample(128, 3)
    assert all(t.device.type == device for t in replay.data.values())
    assert all(t.device.type == device for t in batch.values())
    assert replay.generation.device.type == replay.priority.device.type == device
    torch.testing.assert_close(batch["state"][:, 1:, 0] - batch["state"][:, :-1, 0],
                               torch.ones(128, 2, device=device))
    assert (batch["state"][:, :, 0] >= 2).all()
    expected_mask = torch.cat((torch.ones(128, 1, device=device),
                               (~batch["done"][:, :-1]).float().cumprod(1)), 1)
    torch.testing.assert_close(batch["mask"], expected_mask)
    starts = (torch.arange(4, device=device) + replay.cursor) % replay.capacity
    weights = replay.priority[starts].flatten().pow(.6)
    expected_probability = .2 / 8 + .8 * replay.priority[batch["priority_slot"], batch["env_id"]].pow(.6) / weights.sum()
    torch.testing.assert_close(batch["sample_probability"], expected_probability)
    torch.testing.assert_close(batch["importance_weight"], 1 / (8 * expected_probability))
    assert replay.archive_size == 0
    one = replay.sample(19)
    assert "importance_weight" not in one
    torch.testing.assert_close(one["history"][:, 0, 0], one["state"][:, 0])
    torch.testing.assert_close(one["sensor/offset"][:, 0], one["state"][:, 0])
    real = replay.sample_real(19)
    assert set(real) == {"state", "next_state", "reward", "terminated", "env_id"}
    torch.testing.assert_close(real["next_state"][:, 55:], real["state"][:, 55:])
    torch.testing.assert_close(real["reward"], real["state"][:, 0])
    for t in range(8, 14):
        replay.add(transition(t, device=device))
    before = replay.priority.clone()
    assert replay.update_priorities(batch, torch.ones(128, device=device) * 1000) == 0
    torch.testing.assert_close(replay.priority, before, atol=0, rtol=0)


def fixed_priority_batch(replay, device):
    slots = torch.tensor([1, 1, 2, 3, 4, 2], device=device)
    generation = replay.generation[slots].clone()
    generation[-2] -= 1
    return dict(priority_slot=slots, env_id=torch.tensor([0, 0, 1, 0, 1, 1], device=device),
                priority_generation=generation)


def test_fixed_size_priority_reducer_matches_cpu_oracle():
    # Exercise the CUDA algorithm using CPU tensors as well; this checks the
    # duplicate/stale/epsilon math even when no GPU has been scheduled.
    reference = Replay(6, 2, "cpu", event_window=3, adaptive=True)
    resident = Replay(6, 2, "cpu", event_window=3, adaptive=True)
    for t in range(6):
        reference.add(transition(t))
        resident.add(transition(t))
    batch = fixed_priority_batch(reference, "cpu")
    for errors in (torch.tensor([.25, 2., 2e6, 0., 7e6, 7.]),
                   torch.tensor([.1, .2, .3, .4, .5, .6]), torch.zeros(6)):
        expected = reference.update_priorities(batch, errors)
        actual = resident._update_priorities_resident(errors, batch["priority_slot"],
                                                       batch["env_id"], batch["priority_generation"])
        assert actual == expected == 3
        torch.testing.assert_close(resident.priority, reference.priority, atol=0, rtol=0)
        assert resident.max_priority == reference.max_priority
    # The maximum is historical, but previously sampled cells may decrease.
    assert resident.priority[1, 0] == torch.tensor(1e-5)
    assert resident.max_priority == 1e6
    for t in range(6, 12):
        reference.add(transition(t))
        resident.add(transition(t))
    stale = resident._update_priorities_resident(torch.full((6,), 1e8), batch["priority_slot"],
                                                 batch["env_id"], batch["priority_generation"])
    assert stale == reference.update_priorities(batch, torch.full((6,), 1e8)) == 0
    torch.testing.assert_close(resident.priority, reference.priority, atol=0, rtol=0)


@CUDA
def test_cuda_fixed_tokens_match_cpu_and_reject_bad_errors():
    cpu = Replay(6, 2, "cpu", event_window=3, adaptive=True)
    gpu = Replay(6, 2, "cuda", event_window=3, adaptive=True, storage_device="cuda")
    for t in range(6):
        cpu.add(transition(t))
        gpu.add(transition(t, device="cuda"))
    batch = fixed_priority_batch(cpu, "cpu")
    gpu_batch = {k: v.cuda() for k, v in batch.items()}
    for errors in (torch.tensor([.25, 2., 2e6, 0., 7e6, 7.]), torch.zeros(6)):
        count = gpu.update_priorities(gpu_batch, errors.cuda())
        assert count.device.type == "cuda"
        assert count == cpu.update_priorities(batch, errors)
        torch.testing.assert_close(gpu.priority.cpu(), cpu.priority, atol=0, rtol=0)
        assert gpu.max_priority == cpu.max_priority
    for value in (float("nan"), float("inf"), -1.):
        with pytest.raises(ValueError, match="finite nonnegative"):
            gpu.update_priorities(gpu_batch, torch.full((6,), value, device="cuda"))
    with pytest.raises(ValueError, match="One priority"):
        gpu.update_priorities(gpu_batch, torch.ones(7, device="cuda"))


@CUDA
def test_cuda_importance_weight_corrects_bias():
    torch.cuda.manual_seed_all(20)
    replay = Replay(5, 2, "cuda", event_window=2, adaptive=True, storage_device="cuda")
    for t in range(5):
        replay.add(transition(t, device="cuda"))
    replay.priority[0] = 1000
    batch = replay.sample(60000, 2)
    p = batch["sample_probability"]
    assert (p >= .2 / 8 - 1e-8).all()
    torch.testing.assert_close(batch["importance_weight"], 1 / (8 * p))
    assert batch["state"][:, 0, 0].mean() < .6
    corrected = (batch["state"][:, 0, 0] * batch["importance_weight"]).mean()
    assert abs(float(corrected) - 1.5) < .05


@pytest.mark.parametrize("device", DEVICES)
def test_device_event_archive_keeps_terminal_excludes_reset(device):
    replay = Replay(16, 2, device, event_capacity=12, event_fraction=1,
                    storage_device=device)
    for t in range(8):
        replay.add(transition(t, t == 7, device))
    assert replay.archive_size == 2
    assert all(t.device.type == device for t in replay.archive.values())
    batch = replay.sample(10, 8)
    assert batch["done"][:, -1].all()
    assert not batch["done"][:, :-1].any()
    assert (batch["mask"] == 1).all()
    for t in range(8, 15):
        replay.add(transition(t, device=device))
    assert replay.archive_size == 2
    replay.add(transition(15, device=device))
    assert replay.archive_size == 4
    assert replay.archive["action"][2, 0, 0] == 8


def test_unsupported_storage_fails_without_allocation():
    with pytest.raises(ValueError, match="CPU or CUDA"):
        Replay(6, 2, "cpu", storage_device="meta")
