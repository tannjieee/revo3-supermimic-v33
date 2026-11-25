"""Two-process checks against pooled batches, plus real SuperMimic optimizer steps."""

import copy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from viserdex.supermimic.distributed import Distributed
from viserdex.supermimic.networks import RunningNorm
from viserdex.supermimic.trainer import Trainer, checked_step
from test_supermimic import config, make_replay


def distributed_checks(device, output):
    d = Distributed(device, enabled=True)
    rank = d.rank
    torch.manual_seed(123)
    x = torch.randn(10, 3, device=device) * 3 + 10
    y = torch.randn(10, 2, device=device)
    sl = slice(0, 3) if rank == 0 else slice(3, 10)
    norm, reference = RunningNorm(3).to(device), RunningNorm(3).to(device)
    for shift in (0, 0.5):
        norm.update(x[sl] + shift, d)
        reference.update(x + shift)
        for a, b in zip(norm.buffers(), reference.buffers()):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)

    # Unequal valid counts must match pooling BEFORE clipping and Adam.
    model = torch.nn.Linear(3, 2).to(device)
    pooled = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pooled_optimizer = torch.optim.Adam(pooled.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        pooled_optimizer.zero_grad()
        local_loss = (model(x[sl]) - y[sl]).square().sum() / d.batch_denominator(len(x[sl]))
        local_loss.backward()
        ((pooled(x) - y).square().sum() / len(x)).backward()
        checked_step(model, optimizer, True, d)
        checked_step(pooled, pooled_optimizer, True)
        for a, b in zip(model.parameters(), pooled.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
        d.assert_models_equal(model)

    cfg = config()
    replay, state, limits, noise = make_replay(cfg)
    replay.device = str(device)
    state[:, 42] += rank * 0.02
    replay.data["state"][:, :, 42] += rank * 0.02
    replay.data["next_physical"][:, :, 42] += 0.01 + rank * 0.03
    state, limits = state.to(device), limits.to(device)
    context = torch.full((6, 2), 1.0 + rank, device=device)
    base = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).expand(6, -1)
    torch.manual_seed(123 + rank)
    trainer = Trainer(cfg, noise, context, limits, base, str(device), distributed=d)
    trainer.update_real_statistics(
        state, torch.zeros(6, 21, device=device), state[:, :55], replay.data["history"][0, :, 0].to(device)
    )
    assert trainer.actor.obs_norm.count == 12
    for _ in range(2):
        metrics = trainer.update(replay)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        digest = d.assert_models_equal(trainer.wm, trainer.actor, trainer.critic)
    assert trainer.optimizer_steps == dict(wm=2, actor=2, critic=12)
    metadata = dict(environment_contract="test", joint_names=[], context_schema={}, observation_protocol="test")
    if rank == 0:
        trainer.save(Path(output) / "test.pt", metadata)
    d.barrier()
    restored = Trainer(cfg, noise, context, limits, base, str(device), distributed=d)
    restored.restore(
        torch.load(Path(output) / "test.pt", map_location="cpu", weights_only=False), metadata, restore_rng=False
    )
    assert restored.optimizer_steps == trainer.optimizer_steps
    assert d.assert_models_equal(restored.wm, restored.actor, restored.critic) == digest
    trainer.update(replay)
    next_digest = d.assert_models_equal(trainer.wm, trainer.actor, trainer.critic)
    assert next_digest != digest
    aggregate = d.metrics(
        dict(
            terminated=1 + 2 * rank,
            timeouts=0,
            completed_episode_goals=2.0 + 7 * rank,
            reward_sum=float(rank),
            successes=rank,
            task_samples=4*rank,
            task_position_sum_m=.12*rank,
            task_angle_sum_rad=2.*rank,
            task_stable_samples=rank,
            post_evaluation_samples=2*rank,
            task_mean_position_m=.03 if rank else None,
            task_mean_angle_rad=.5 if rank else None,
            task_stable_fraction=.25 if rank else None,
            parity_max_errors=({"terminal_reset_state_separation": 1.0, "action_second_diff": .125} if rank else {"action_second_diff": .0625}),
        )
    )
    assert aggregate["goals_per_completed_episode"] == 2.75
    assert aggregate["reward_sum"] == 0.5
    assert aggregate["parity_max_errors"]["terminal_reset_state_separation"] == 1
    assert aggregate["parity_max_errors"]["action_second_diff"] == .125
    assert aggregate['task_samples'] == 4
    assert aggregate['task_mean_position_m'] == .03
    assert aggregate['task_mean_angle_rad'] == .5
    assert aggregate['task_stable_fraction'] == .25
    assert aggregate['post_evaluation_samples'] == 2
    # Current orientation-only runs report hand-center distance and five cost
    # heads. Pool their counts/totals, preserve maxima, and derive means from
    # the pooled denominator (including a rank with no samples).
    for samples in ((3, 7), (0, 7)):
        n = samples[rank]
        mean_distance = .1 + .2 * rank
        gate_mean = .2 + .4 * rank
        local = dict(task_samples=n, task_hand_center_distance_sum_m=n * mean_distance,
                     task_mean_hand_center_distance_m=mean_distance if n else None,
                     object_hold_cost_gate_sum=n * gate_mean,
                     object_hold_cost_gate_mean=gate_mean if n else None)
        names = ('joint_speed', 'target_rate', 'object_angular_speed', 'object_linear_speed', 'drop')
        for i, name in enumerate(names, 1):
            local[f'cost_{name}_sum'] = float(i * (rank + 1))
            local[f'cost_{name}_violations'] = i * (rank + 1)
            local[f'cost_{name}_max_ratio'] = float(i + rank)
        aggregate = d.metrics(local)
        assert aggregate['task_samples'] == sum(samples)
        torch.testing.assert_close(torch.tensor(aggregate['task_hand_center_distance_sum_m']),
                                   torch.tensor(samples[0] * .1 + samples[1] * .3))
        torch.testing.assert_close(torch.tensor(aggregate['task_mean_hand_center_distance_m']),
                                   torch.tensor((samples[0] * .1 + samples[1] * .3) / sum(samples)))
        torch.testing.assert_close(torch.tensor(aggregate['object_hold_cost_gate_mean']),
                                   torch.tensor((samples[0] * .2 + samples[1] * .6) / sum(samples)))
        for i, name in enumerate(names, 1):
            assert aggregate[f'cost_{name}_sum'] == 3. * i
            assert aggregate[f'cost_{name}_violations'] == 3 * i
            assert aggregate[f'cost_{name}_max_ratio'] == i + 1.
        # Console formatting is an independent optional workstream. Exercise
        # the integration when present without requiring its untracked module.
        if importlib.util.find_spec('viserdex.supermimic.console_log') is not None:
            from viserdex.supermimic.console_log import format_training_log
            aggregate.update(iteration=1, total_num_envs=sum(samples), world_size=2)
            formatted = format_training_log(aggregate, total_iterations=10, collection_steps=1)
            expected_rate = f"{100 * aggregate['cost_joint_speed_violations'] / sum(samples):.2f} %"
            assert f"{'Joint speed violations':>30}: {expected_rate}" in formatted
    assert d.any(rank == 1)
    Path(output, f"rank_{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                digest=next_digest,
                optimizer_steps=trainer.optimizer_steps,
                normalization_count=float(trainer.actor.obs_norm.count),
            )
        )
    )


def cpu_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        distributed_checks(torch.device("cpu"), output)
    finally:
        dist.destroy_process_group()


def test_two_rank_gradients_normalizers_and_supermimic_updates(tmp_path):
    mp.spawn(cpu_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["digest"] == results[1]["digest"]
