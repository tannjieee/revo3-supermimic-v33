"""Numerical failure evidence must distinguish norm overflow from NaN entries."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/supermimic'))
from experiment_contact_sdf import gradient_failure_details


def test_finite_float32_gradients_can_overflow_norm_without_becoming_nan():
    model = torch.nn.Linear(3, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 1e25)
    before = model.weight.grad.clone()
    with pytest.raises(RuntimeError, match='non-finite'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    detail = gradient_failure_details(model)['weight']
    assert detail['all_entries_finite']
    assert detail['norm_float64'] == pytest.approx(3**.5 * 1e25)
    torch.testing.assert_close(before, model.weight.grad)
    json.dumps(detail, allow_nan=False)


def test_nan_gradient_is_recorded_without_nonstandard_json_numbers():
    model = torch.nn.Linear(3, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    detail = gradient_failure_details(model)['weight']
    assert not detail['all_entries_finite']
    assert detail['norm_float64'] == 'nan'
    json.dumps(detail, allow_nan=False)


def test_analysis_preserves_failed_arms_and_uses_only_available_pairs(tmp_path, monkeypatch):
    import analyze_contact_sdf as analysis

    fits = tmp_path / 'fits'
    fits.mkdir()
    (tmp_path / 'plan.json').write_text(json.dumps({'test_seeds': [63]}))
    rotation = {'resolved': 0, 'base_rmse': 0., 'quiet': {'count': 2, 'false_response_gt1e_4': 0}}
    successful = {'complete': True, 'test': {'all': {'8': {'rotation': rotation}}}}
    result = {'status': 'complete_with_failed_fits', 'sampling_seeds': [75, 76], 'fits': {
        'baseline_seed75': successful, 'quasi_seed75': successful,
        'dynamic_seed75': {'failed': True, 'complete': False, 'failure_update': 3,
                           'failure': 'non-finite norm', 'gradient_failure_details': {}},
    }}
    (fits / 'result.json').write_text(json.dumps(result))
    base = torch.zeros(8, 2, 55)
    base[..., 48] = 1.
    plus = base.clone()
    plus[..., 0] = .1
    for arm in ('baseline', 'quasi'):
        torch.save({'base': base, 'minus': base, 'plus': plus}, fits / f'{arm}_seed75_test_predictions.pt')
    response = torch.zeros(8, 2, 54)
    response[..., 0] = .1
    resolved = torch.zeros(8, 2, 6, dtype=torch.bool)
    resolved[..., 0] = True
    monkeypatch.setattr(analysis, 'dataset', lambda *args: {
        'cluster': torch.tensor([1, 2]), 'response': response, 'resolved': resolved,
    })
    monkeypatch.setattr(sys, 'argv', ['analysis', '--experiment', str(tmp_path), '--fits', str(fits)])
    analysis.main()
    report = json.loads((fits / 'comparison.json').read_text())
    assert len(report['paired_comparisons']['q/quasi']['per_seed']) == 1
    assert report['paired_comparisons']['q/dynamic']['status'] == 'no_completed_pair'
    assert report['failed_fits']['dynamic_seed75']['update'] == 3
    assert not report['policy_training_promoted']
