"""Independent task-loss, terminal masking, gradient, calibration and version checks."""
from dataclasses import asdict, replace
import copy
import math
import pytest
import torch
from viserdex.supermimic.core import Config, advance, action_target, rotation_vector_quat, POS, ROT
from viserdex.supermimic.trainer import Trainer, checkpoint_version, frozen
from viserdex.supermimic.networks import Dynamics
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.sensors import initial_sensors, observe
from viserdex.supermimic.wm_supervision import clean_next_observation, task_losses, task_reward
from viserdex.supermimic.termination_calibration import binary_metrics, fit_affine_calibration, rollout_samples, FrozenTerminationCalibration
from viserdex.distillation.config import PerceptionNoiseCfg
from test_hand_center_contract import fixture


def setup(n=4, device='cpu', **kwargs):
    cfg, state, limits = fixture(n)
    cfg = replace(cfg, wm_task_losses=True, wm_hidden=(16,), actor_hidden=(16,), critic_hidden=(16,),
                  wm_batch=n, wm_window=3, actor_batch=n, horizon=3, microbatch=n,
                  wm_microbatch=n, actor_microbatch=n, critic_epochs=1, critic_minibatches=1, **kwargs)
    state, limits = state.to(device), limits.to(device)
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False, joint_delay_max_steps=0)
    trainer = Trainer(cfg, noise, torch.ones(n, 65, device=device), limits,
                      torch.tensor([[0., 0., 0., 1.]], device=device).expand(n, -1), device)
    return trainer, state, limits


def window(trainer, state):
    n, k = len(state), trainer.cfg.wm_window
    return dict(state=state[:, None].expand(-1, k, -1).clone(),
                next_physical=state[:, None, :55].expand(-1, k, -1).clone(),
                action=torch.zeros(n, k, 21, device=state.device),
                reward=torch.full((n, k), .4, device=state.device),
                terminated=torch.zeros(n, k, dtype=torch.bool, device=state.device),
                done=torch.zeros(n, k, dtype=torch.bool, device=state.device),
                mask=torch.ones(n, k, device=state.device), env_id=torch.arange(n, device=state.device))


def test_clean_target_matches_noise_disabled_policy_with_actual_last_action():
    tr, state, limits = setup()
    nxt = state[:, :55].clone()
    nxt[:, 21:42] = .7
    nxt[:, POS] += .015
    action = torch.full((4, 21), .2)
    target = action_target(state, action, limits, tr.cfg)
    full = advance(state, nxt, action, target, state[:, 79:83], hold_cfg=tr.cfg)
    expected, _ = observe(full, initial_sensors(4, 'cpu', tr.noise_cfg), limits, tr.noise_cfg, tr.cfg)
    actual = clean_next_observation(state, nxt, action, limits, tr.cfg)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, 42:63], action)


def test_perfect_state_has_zero_task_consistency_losses_and_real_bce():
    tr, state, limits = setup()
    action = torch.zeros(4, 21)
    nxt = state[:, :55].clone()
    actual = task_reward(state, nxt, action, tr.cfg)
    labels = torch.tensor([False, True, False, True])
    total, components = task_losses(tr, state, nxt, state, nxt, action, limits, actual, labels)
    assert torch.equal(components[:, [0, 1, 3, 4, 5, 6, 7, 8]], torch.zeros(4, 8))
    expected = torch.tensor([-math.log(.99), -math.log(.01)] * 2)
    torch.testing.assert_close(total, expected, atol=1e-6, rtol=1e-6)
    total.mean().backward()
    assert tr.wm.termination_net[-1].bias.grad.abs().sum() > 0


def test_reward_observation_and_cost_losses_supply_physical_gradients():
    tr, state, limits = setup()
    action = torch.zeros(4, 21)
    pred = state[:, :55].clone()
    pred[:, ROT] = rotation_vector_quat(torch.tensor([[.3, 0., 0.]]).expand(4, -1))
    pred[:, POS.start] += .35
    pred[:, 21] = math.pi
    pred.requires_grad_()
    _, components = task_losses(tr, state, pred, state, state[:, :55], action, limits,
                               torch.full((4,), .4), torch.zeros(4, dtype=torch.bool))
    # Joint cost at twice the speed threshold is 1; target is 0, hence log(2)^2.
    torch.testing.assert_close(components[:, 4], torch.full((4,), math.log(2)**2))
    for column in (0, 1, 3):
        gradient, = torch.autograd.grad(components[:, column].sum(), pred, retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_terminal_sample_included_but_reset_and_timeout_not_positive_labels():
    tr, state, _ = setup(2)
    w = window(tr, state)
    w['terminated'][0, 1] = True
    w['done'][:, 1] = True  # World 1 is a time limit, not a failure.
    w['mask'][:, 2] = 0
    _, _, _, _, losses = tr._wm_loss(w)
    expected = (3 * -math.log(.99) - math.log(.01)) / 2
    assert losses[2].item() == pytest.approx(expected, abs=1e-6)
    # Arbitrary *finite* post-reset targets have no effect on loss or gradients.
    before = tr._wm_loss(w)[0]
    w['reward'][:, 2] = 999.
    w['terminated'][:, 2] = True
    w['next_physical'][:, 2, :42] = 100.
    torch.testing.assert_close(tr._wm_loss(w)[0], before)


def test_masked_reset_tail_cannot_poison_loss_or_gradients():
    tr, state, _ = setup(2)
    # Nonzero dynamics Jacobian checks the recurrent backward path too.
    torch.nn.init.normal_(tr.wm.net[-1].weight, std=.01)
    w = window(tr, state)
    w['done'][:, 0] = True
    w['mask'][:, 1:] = 0
    expected = tr._wm_loss(w)[0]
    expected.backward()
    gradients = [p.grad.clone() for p in tr.wm.parameters()]
    tr.opt_wm.zero_grad(set_to_none=True)
    for key in ('state','next_physical','action','reward'):
        w[key][:, 1:] = float('nan')
    actual = tr._wm_loss(w)[0]
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for p, reference in zip(tr.wm.parameters(), gradients):
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, reference, atol=0, rtol=0)


def test_physical_metrics_use_same_importance_weights_as_wm_objective():
    tr, state, _ = setup(2)
    w = window(tr, state)
    w['next_physical'][0, :, 0] += .1
    w['next_physical'][1, :, 0] += .5
    w['importance_weight'] = torch.tensor([.25, 1.75])
    full = tr._wm_loss(w)
    singles = [tr._wm_loss({k:v[i:i+1] for k,v in w.items()}) for i in range(2)]
    torch.testing.assert_close(full[2], singles[0][2]+singles[1][2])
    # With task losses removed, physical metrics exactly reconstruct loss.
    from dataclasses import replace
    tr.cfg = replace(tr.cfg, wm_task_losses=False)
    loss, _, physical, _, _ = tr._wm_loss(w)
    widths = physical.new_tensor([21,21,3,3,3,3])
    torch.testing.assert_close(loss, (physical*widths).sum()/widths.sum())


@pytest.mark.parametrize('adaptive', [False, True])
def test_replay_preserves_terminal_labels_and_importance_weights(adaptive):
    tr, state, _ = setup(2)
    w = window(tr, state)
    replay = Replay(6, 2, 'cpu', event_capacity=0, event_window=3, adaptive=adaptive)
    for t in range(3):
        replay.add({k: v[:, t] for k, v in w.items() if k not in ('env_id', 'mask')})
    sample = replay.sample(12, 3, include_events=False)
    assert sample['reward'].shape == sample['terminated'].shape == (12, 3)
    if adaptive:
        torch.testing.assert_close(sample['importance_weight'], torch.ones(12))


def test_survival_uses_head_and_actor_gradients_cross_frozen_head():
    tr, state, limits = setup()
    tr.wm.termination_net[-1].bias.data.zero_()
    torch.testing.assert_close(tr.continuation(state, state[:, :55]), torch.full((4,), .5))
    tr.wm.termination_temperature.fill_(2.)
    tr.wm.termination_bias.fill_(1.)
    torch.testing.assert_close(tr.continuation(state, state[:, :55]), torch.full((4,), 1-torch.sigmoid(torch.tensor(1.)).item()))
    torch.nn.init.normal_(tr.wm.net[-1].weight, std=.01)
    torch.nn.init.normal_(tr.wm.termination_net[-1].weight, std=.01)
    action = torch.zeros(4, 21, requires_grad=True)
    with frozen(tr.wm):
        prediction = tr.wm(state, action_target(state, action, limits, tr.cfg), tr.context)
        tr.continuation(state, prediction).sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    assert all(p.grad is None for p in tr.wm.parameters())


def test_v28_roundtrip_and_old_v27_strict_compatibility(tmp_path):
    tr, state, _ = setup()
    old_cfg = replace(tr.cfg, wm_task_losses=False)
    saved = asdict(old_cfg)
    for name in list(saved):
        if name.startswith('wm_') and ('loss_weight' in name or name == 'wm_task_losses'):
            saved.pop(name)
    old = Dynamics(65, Config(**saved))
    old.load_state_dict(Dynamics(65, old_cfg).state_dict(), strict=True)
    assert checkpoint_version(old_cfg) == 27 and checkpoint_version(tr.cfg) == 28
    with pytest.raises(RuntimeError, match='Missing key'):
        tr.wm.load_state_dict(old.state_dict(), strict=True)
    tr.wm.termination_temperature.fill_(1.7)
    tr.wm.termination_bias.fill_(.2)
    metadata = dict(environment_contract='a', joint_names=[], context_schema={}, observation_protocol='test')
    tr.save(tmp_path/'v28.pt', metadata)
    new, _, _ = setup()
    new.restore(torch.load(tmp_path/'v28.pt', weights_only=False), metadata, restore_rng=False)
    torch.testing.assert_close(new.wm.termination_probability(state[:, :55]), tr.wm.termination_probability(state[:, :55]))


def test_calibration_metrics_and_sparse_labels_are_not_false_success():
    metrics = binary_metrics(torch.zeros(4), torch.tensor([0, 0, 1, 1]))
    assert metrics['nll'] == pytest.approx(math.log(2)) and metrics['brier'] == .25 and metrics['ece'] == 0
    sparse = fit_affine_calibration(torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(100))
    assert not sparse['accepted'] and sparse['temperature'] == 1. and sparse['bias'] == 0.
    assert sparse['reason'].startswith('insufficient')


def test_affine_calibration_improves_distinct_validation_data():
    generator = torch.Generator().manual_seed(731)
    x = torch.randn(6000, generator=generator)
    labels = torch.rand(6000, generator=generator) < x.sigmoid()
    miscalibrated = 3*x + 1.2
    report = fit_affine_calibration(miscalibrated[:3000], labels[:3000], miscalibrated[3000:], labels[3000:])
    assert report['accepted']
    assert report['validation_after']['nll'] < report['validation_before']['nll']
    assert report['validation_after']['brier'] < report['validation_before']['brier']


def test_frozen_calibration_masks_after_done_and_splits_whole_worlds():
    tr, state, limits = setup()
    w = window(tr, state)
    records = [{k: v[:, t] for k, v in w.items() if k not in ('env_id', 'mask')} for t in range(3)]
    records[0]['done'][0] = True
    records[0]['terminated'][1] = True
    records[0]['done'][1] = True
    samples = rollout_samples(tr.wm, records, tr.context, limits, tr.cfg)
    assert int((samples['env_id'] == 0).sum()) == int((samples['env_id'] == 1).sum()) == 1
    assert int(samples['labels'].sum()) == 1
    calibration = FrozenTerminationCalibration(tr.wm, tr.context, limits, tr.cfg)
    for record in records:
        calibration.observe(record)
    report, _ = calibration.finish()
    assert not report['accepted'] and report['test_before']['samples'] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('delay_ready', [False, True])
def test_cuda_capture_matches_eager_with_all_auxiliary_losses(delay_ready):
    options = (dict(include_delay_ready=True,absolute_policy_optimization=True,stable_value_learning=True)
               if delay_ready else {})
    tr, state, _ = setup(device='cuda', **options)
    w = window(tr, state)
    w['next_physical'][:, :, POS.start] += .02
    w['terminated'][0, 2] = True
    w['done'][1, 0] = True
    w['mask'][1, 1:] = 0
    for key in ('state','next_physical','action','reward'):
        w[key][1, 1:] = float('nan')
    eager = copy.deepcopy(tr)
    class Fixed:
        def sample(self, *args, **kwargs):
            return w
        def update_priorities(self, *args):
            return 0
    tr.wm_cuda_graph = True
    a, b = tr.wm_update(Fixed()), eager.wm_update(Fixed())
    for name in a:
        assert a[name] == pytest.approx(b[name], rel=1e-5, abs=1e-6), name
    for name, parameter in tr.wm.state_dict().items():
        torch.testing.assert_close(parameter, eager.wm.state_dict()[name], atol=2e-6, rtol=1e-5)
    assert a['wm_loss_observation'] > 0 and a['wm_loss_termination'] > 0
