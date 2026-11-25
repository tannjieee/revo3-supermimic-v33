"""Human-readable training progress; full metrics remain in metrics.jsonl."""
import math


def format_training_log(metrics, total_iterations, collection_steps):
    """Format existing metrics without changing training or episode statistics."""
    def number(key, precision=4, scale=1.):
        value = metrics.get(key)
        return '--' if value is None else f'{value * scale:.{precision}f}'

    def row(label, value):
        return f'{label:>30}: {value}'

    iteration = metrics['iteration']
    phase = metrics.get('phase', 'unknown')
    phase_label = {'wm_critic_warmup': 'WM + Critic warmup (Actor frozen)',
                   'wm_warmup': 'WM warmup (Actor frozen)',
                   'joint': 'Joint training'}.get(phase, phase)
    collection = metrics.get('collection_seconds', 0.)
    update = metrics.get('update_seconds', 0.)
    num_envs = metrics.get('total_num_envs', 0)
    transitions = collection_steps * num_envs
    throughput = transitions / (collection + update) if collection + update > 0 else 0.
    reward = (f"{metrics['reward_sum'] / collection_steps:.5f}"
              if collection_steps and metrics.get('reward_sum') is not None else '--')
    actor_loss = number('actor_loss') if phase == 'joint' else 'frozen'
    lines = [
        '#' * 80,
        f' Learning iteration {iteration}/{total_iterations} '.center(80),
        row('Phase', phase_label),
        row('Throughput (collect + update)', f'{throughput:,.0f} transitions/s'),
        row('Time (collect / update)', f'{collection:.3f}s / {update:.3f}s'),
        '-' * 80,
        row('Loss (WM / Critic / Actor)', f"{number('wm_loss')} / {number('critic_loss')} / {actor_loss}"),
        row('Actor LR / gradient norm', f"{number('actor_learning_rate', 7)} / {number('actor_grad_norm')}"),
        row('Mean reward / env-step', reward),
        row('Mean orientation error', f"{number('task_mean_angle_rad', 2, 180 / math.pi)} deg"),
    ]
    distance = ('task_mean_hand_center_distance_m' if 'task_mean_hand_center_distance_m' in metrics
                else 'task_mean_position_m')
    label = 'Mean hand-center distance' if 'hand_center' in distance else 'Mean position error'
    lines += [row(label, f'{number(distance, 2, 100)} cm'),
              row('Stable samples', f"{number('task_stable_fraction', 2, 100)} %"),
              row('Success / drop / timeout', ' / '.join(str(metrics.get(k, '--'))
                  for k in ('successes', 'drops', 'timeouts')))]
    # Distributed.metrics sums violation counts across ranks, matching the
    # global transition count used as the percentage denominator.
    for key, label in (('joint_speed', 'Joint speed violations'),
                       ('target_rate', 'Target rate violations'),
                       ('object_angular_speed', 'Object angular violations'),
                       ('object_linear_speed', 'Object linear violations')):
        count = metrics.get(f'cost_{key}_violations')
        if count is not None:
            rate = f'{100 * count / transitions:.2f} %' if transitions else '--'
            lines.append(row(label, rate))
    if 'actor_gradient_cost_reward_ratio' in metrics:
        lines.append(row('Cost / reward gradient', number('actor_gradient_cost_reward_ratio', 3)))
    steps = metrics.get('optimizer_steps', {})
    elapsed = int(metrics.get('elapsed_seconds', 0.))
    lines += [
        '-' * 80,
        row('Updates (WM / Critic / Actor)', ' / '.join(str(steps.get(k, '--')) for k in ('wm', 'critic', 'actor'))),
        row('Transitions since restart', f"{metrics.get('real_transitions_since_restart', 0):,}"),
        row('GPU peak / replay', f"{number('gpu_peak_gib', 2)} / {number('replay_allocated_gib', 2)} GiB"),
        row('Elapsed', f'{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}'),
        ' Collection statistics cover this iteration; success/drop/timeout are event counts.',
        '#' * 80,
    ]
    return '\n'.join(lines)
