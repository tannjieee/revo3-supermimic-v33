"""Report fixed-budget ContactSDF fits after all held-out predictions exist.

Resample physical worlds, retaining repeated starts within each world. The two
sampling seeds are not independent end-to-end training replications. This script
never selects checkpoints or promotes a model into policy training.
"""
import argparse
import json
from pathlib import Path

import torch

from analyze_contact_features import cluster_ci
from experiment_contact_features import GROUPS, SIZES, dataset, physical_delta, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--fits', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = json.loads((args.fits / 'result.json').read_text())
    if result['status'] not in ('complete', 'complete_with_failed_fits'):
        raise ValueError('Analysis requires every prespecified fit and held-out prediction')
    plan = json.loads((args.experiment / 'plan.json').read_text())
    test = dataset([args.experiment / f'branches_seed{s}' for s in plan['test_seeds']], 'cpu')
    cluster = test['cluster']
    predictions = {
        key: torch.load(args.fits / f'{key}_test_predictions.pt', weights_only=False)
        for key, entry in result['fits'].items() if entry.get('complete')
    }
    responses = {key: physical_delta(p['minus'], p['plus']) for key, p in predictions.items()}
    report = {
        'scope': 'Fixed final update; world-cluster intervals; minibatch seeds only; no automatic promotion',
        'source_result_sha256': sha(args.fits / 'result.json'),
        'test_worlds': int(cluster.unique().numel()),
        'test_starts': len(cluster),
        'physics_response_concentration': {},
        'paired_comparisons': {},
        'summary': {key: entry['test'] for key, entry in result['fits'].items() if entry.get('complete')},
        'failed_fits': {key: {'update': entry['failure_update'], 'reason': entry['failure'],
                             'gradient_details': entry['gradient_failure_details']}
                        for key, entry in result['fits'].items() if entry.get('failed')},
        'policy_training_promoted': False,
    }
    for g, (name, real) in enumerate(zip(GROUPS, test['response'][7].split(SIZES, -1))):
        mask = test['resolved'][7, :, g]
        if not mask.any():
            report['physics_response_concentration'][name] = {'resolved': 0, 'worlds': 0}
            continue
        truth = real[mask]
        norm = truth.norm(dim=-1)
        energy = truth.square().sum(-1)
        report['physics_response_concentration'][name] = {
            'resolved': int(mask.sum()), 'worlds': int(cluster[mask].unique().numel()),
            'largest_squared_energy_fraction': float(energy.max() / energy.sum()),
        }
        for arm in ('quasi', 'dynamic'):
            cosine_differences, error_differences, per_seed = [], [], []
            for seed in result['sampling_seeds']:
                if f'baseline_seed{seed}' not in responses or f'{arm}_seed{seed}' not in responses:
                    continue
                base = responses[f'baseline_seed{seed}'][7].split(SIZES, -1)[g][mask]
                alt = responses[f'{arm}_seed{seed}'][7].split(SIZES, -1)[g][mask]
                base_cos = (base * truth).sum(-1) / (base.norm(dim=-1) * norm).clamp_min(1e-20)
                alt_cos = (alt * truth).sum(-1) / (alt.norm(dim=-1) * norm).clamp_min(1e-20)
                difference = alt_cos - base_cos
                cosine_differences.append(difference)
                error_differences.append(((alt - truth).norm(dim=-1) - (base - truth).norm(dim=-1)) / norm)
                per_seed.append({
                    'seed': seed, 'cosine_difference': float(difference.mean()),
                    'cosine_difference_world95': cluster_ci(difference, cluster[mask]),
                    'baseline_relative_vector_error': float((base - truth).norm() / truth.norm()),
                    'alternative_relative_vector_error': float((alt - truth).norm() / truth.norm()),
                })
            if not cosine_differences:
                report['paired_comparisons'][f'{name}/{arm}'] = {'status': 'no_completed_pair'}
                continue
            cosine = torch.stack(cosine_differences).mean(0)
            error = torch.stack(error_differences).mean(0)
            report['paired_comparisons'][f'{name}/{arm}'] = {
                'per_seed': per_seed,
                'cosine_difference': float(cosine.mean()),
                'cosine_difference_world95': cluster_ci(cosine, cluster[mask]),
                'relative_error_difference': float(error.mean()),
                'relative_error_difference_world95': cluster_ci(error, cluster[mask]),
            }
    save_json(args.fits / 'comparison.json', report)
    lines = [
        '# ContactSDF：固定预算独立测试', '',
        f"测试 {report['test_worlds']} 个物理域、{report['test_starts']} 个起点；两个 seed 只改变 minibatch 顺序。",
        '', '| 模型 | 旋转可辨数 | 旋转方向余弦 | 幅值比中位数 | 相对向量误差 | 基线轨迹旋转 RMSE(rad) | 静止误报 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for key, summary in report['summary'].items():
        value = summary['all']['8']['rotation']
        quiet = value['quiet']
        amplitude = value.get('amplitude_ratio') or {}
        lines.append(
            f"| {key} | {value['resolved']} | {value.get('cosine_mean', float('nan')):.4f} | "
            f"{amplitude.get('median', float('nan')):.3f} | {value.get('relative_vector_error', float('nan')):.3f} | "
            f"{value['base_rmse'] if value['base_rmse'] is not None else float('nan'):.5f} | "
            f"{quiet['false_response_gt1e_4']}/{quiet['count']} |"
        )
    for key, failure in report['failed_fits'].items():
        lines += ['', f"{key} 在第 {failure['update']} 次更新触发数值保护，未达到固定预算；不计算最终模型分数，也不将其隐去。"]
    lines += ['',
        '方向余弦只描述真实响应超过重复噪声阈值的条件子集；不能忽略幅值、无响应误报和绝对轨迹误差。',
        'comparison.json 同时保留 1/2/4/8 步、脉冲/保持时长、不同幅值和六类物理量；区间按物理域重采样。',
        '零响应预测的相对向量误差为 1。参数梯度正确不证明动力学准确；本实验未更新 Actor/Critic，也未自动推广模型。',
    ]
    (args.fits / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
