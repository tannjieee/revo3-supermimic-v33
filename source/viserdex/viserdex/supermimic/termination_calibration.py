"""Frozen-model termination diagnostics and separately validated calibration.

Raw, naturally sampled labels only: no positive-class reweighting. A calibrated
model must stay frozen; Trainer resets this calibration before any WM refit.
"""
import math
import torch
from torch.nn import functional as F
from .core import action_target, advance
from .state_layout import COMPACT
from .spatial_contract import drop_distance


def binary_metrics(logits, labels, bins=10):
    logits, labels = logits.detach().double().flatten(), labels.detach().double().flatten()
    if logits.shape != labels.shape or not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Termination diagnostics require finite logits and hard 0/1 labels')
    n = len(labels)
    if not n:
        return dict(samples=0, positives=0, nll=None, brier=None, ece=None)
    probabilities = logits.sigmoid()
    positive, predicted = labels.bool(), probabilities >= .5
    calibration_error = logits.new_zeros(())
    bucket = (probabilities * bins).long().clamp_max(bins - 1)
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            calibration_error += mask.double().mean() * (probabilities[mask].mean() - labels[mask].mean()).abs()
    return dict(samples=n, positives=int(positive.sum()), prevalence=float(labels.mean()),
                mean_probability=float(probabilities.mean()),
                nll=float(F.binary_cross_entropy_with_logits(logits, labels)),
                brier=float((probabilities - labels).square().mean()), ece=float(calibration_error),
                positive_brier=float((probabilities[positive] - 1).square().mean()) if positive.any() else None,
                negative_brier=float(probabilities[~positive].square().mean()) if (~positive).any() else None,
                recall_at_half=float(predicted[positive].double().mean()) if positive.any() else None,
                false_positive_rate_at_half=float(predicted[~positive].double().mean()) if (~positive).any() else None)


def fit_affine_calibration(fit_logits, fit_labels, validation_logits, validation_labels, min_positives=20):
    """Fit p=sigmoid(logit/T+b); validation chooses between fitted and identity.

    Selection metrics are not an unbiased final test. The collection runner
    additionally reports a third, disjoint world partition without using it to fit
    or select. Sparse-class data leaves identity unchanged with an explicit reason.
    """
    x, y, vx, vy = [v.detach().double().cpu().flatten() for v in
                     (fit_logits, fit_labels, validation_logits, validation_labels)]
    baseline_fit, baseline_validation = binary_metrics(x, y), binary_metrics(vx, vy)
    result = dict(accepted=False, temperature=1., bias=0., fit_before=baseline_fit,
                  validation_before=baseline_validation)
    for name, data in (('fit', baseline_fit), ('validation', baseline_validation)):
        if data['positives'] < min_positives or data['samples'] - data['positives'] < min_positives:
            return dict(result, reason=f'insufficient_positive_or_negative_labels_{name}')
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], lr=.5, max_iter=60, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(.25), math.log(4.)).exp()
        loss = F.binary_cross_entropy_with_logits(x / temperature + bias.clamp(-5., 5.), y)
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(math.log(.25), math.log(4.)).exp())
    offset = float(bias.detach().clamp(-5., 5.))
    fit_after = binary_metrics(x / temperature + offset, y)
    validation_after = binary_metrics(vx / temperature + offset, vy)
    accepted = (validation_after['nll'] < baseline_validation['nll'] - 1e-6 and
                validation_after['brier'] <= baseline_validation['brier'] + 1e-8)
    return dict(result, accepted=accepted,
                reason='validation_nll_improved_and_brier_not_worse' if accepted else 'no_validation_improvement',
                temperature=temperature if accepted else 1., bias=offset if accepted else 0.,
                candidate_temperature=temperature, candidate_bias=offset,
                fit_after=fit_after, validation_after=validation_after)


@torch.no_grad()
def rollout_samples(model, records, context, limits, cfg):
    state = records[0]['state']
    valid = torch.ones(len(state), device=state.device, dtype=torch.bool)
    samples = []
    ids = torch.arange(len(state), device=state.device)
    for horizon, row in enumerate(records, 1):
        target = action_target(state, row['action'], limits, cfg)
        prediction = model(state, target, context)
        samples.append(dict(logits=model.termination_logits(prediction)[valid].cpu(),
                            labels=row['terminated'][valid].cpu(), env_id=ids[valid].cpu(),
                            horizon=torch.full_like(ids[valid], horizon).cpu(),
                            analytic_logits=((drop_distance(prediction, cfg) - cfg.drop_radius_m) /
                                             cfg.survival_width)[valid].cpu()))
        # The terminating transition is included. Neither a drop nor a timeout
        # may expose reset-state targets to this autoregressive trajectory.
        valid = valid & ~row['done']
        state = advance(state, prediction, row['action'], target, state[:, COMPACT.goal_rot], hold_cfg=cfg)
    return {key: torch.cat([row[key] for row in samples]) for key in samples[0]}


class FrozenTerminationCalibration:
    def __init__(self, model, context, limits, cfg):
        if not cfg.wm_task_losses or len(context) < 4:
            raise ValueError('Calibration requires a v28 model and at least four environments')
        self.model, self.context, self.limits, self.cfg = model, context, limits, cfg
        self.records, self.samples = [], []

    def observe(self, record):
        self.records.append({k: record[k].detach().clone() for k in
                             ('state', 'action', 'next_physical', 'terminated', 'done')})
        if len(self.records) == self.cfg.wm_window:
            self._flush()

    def _flush(self):
        if self.records:
            self.samples.append(rollout_samples(self.model, self.records, self.context, self.limits, self.cfg))
            self.records.clear()

    def finish(self):
        self._flush()
        if not self.samples:
            raise ValueError('Calibration requires fresh rollout data')
        data = {k: torch.cat([x[k] for x in self.samples]) for k in self.samples[0]}
        partition = data['env_id'] % 4
        fit, selection, test = partition < 2, partition == 2, partition == 3
        result = fit_affine_calibration(data['logits'][fit], data['labels'][fit],
                                       data['logits'][selection], data['labels'][selection])
        result['split'] = 'env_id % 4: 0/1 fit, 2 validation selection, 3 untouched test; all horizons of a world stay together'
        result['protocol'] = 'frozen_wm_fresh_pre_reset_windows_v1'
        result['test_before'] = binary_metrics(data['logits'][test], data['labels'][test])
        result['test_after'] = binary_metrics(data['logits'][test] / result['temperature'] + result['bias'],
                                             data['labels'][test])
        result['test_analytic_baseline'] = binary_metrics(data['analytic_logits'][test], data['labels'][test])
        result['test_by_horizon'] = {}
        for h in range(1, self.cfg.wm_window + 1):
            mask = test & (data['horizon'] == h)
            result['test_by_horizon'][str(h)] = binary_metrics(
                data['logits'][mask] / result['temperature'] + result['bias'], data['labels'][mask])
        result['test_has_both_classes'] = 0 < result['test_before']['positives'] < result['test_before']['samples']
        return result, data
