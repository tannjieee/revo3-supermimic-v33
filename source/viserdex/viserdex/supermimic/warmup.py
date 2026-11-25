"""Real-transition grasp warmup; policy resets and goals remain independent."""
import math
import numpy as np
import torch


class WarmupGraspResetManager:
    """Temporary training-split resets with an explicitly recorded behavior policy.

    No actuator memory is preloaded. The certified hold action is sent through
    the ordinary action path; exploratory offsets begin after the cold-reset
    settling period. Goals retain the normal task's randomized distribution.
    """
    def __init__(self, adapter, bank, seed=0, exploration=.03, settle_steps=40, refresh_steps=4):
        if bank.split != 'train':
            raise ValueError('Warmup must use the training split only')
        if (not 0 <= exploration <= .2 or type(settle_steps) is not int or settle_steps < 0
                or type(refresh_steps) is not int or refresh_steps < 1):
            raise ValueError('Invalid grasp warmup behavior settings')
        if getattr(adapter.env, 'grasp_reset_manager', None) is not None:
            raise ValueError('Warmup cannot replace another grasp reset manager')
        self.adapter, self.bank = adapter, bank
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device=adapter.env.device).manual_seed(seed)
        self.exploration, self.settle_steps, self.refresh_steps = exploration, settle_steps, refresh_steps
        self.ids = np.zeros(adapter.env.num_envs, dtype=int)
        self.counts = np.zeros(len(bank.records), dtype=int)
        self.age = torch.zeros(adapter.env.num_envs, device=adapter.env.device, dtype=torch.long)
        self.hold = torch.zeros(adapter.env.num_envs, 21, device=adapter.env.device)
        self.offset = torch.zeros_like(self.hold)
        adapter.env.grasp_reset_manager = self

    @torch.no_grad()
    def reset(self, env_ids):
        from .grasp_runtime import write_grasp_states
        ids = (torch.arange(self.adapter.env.num_envs, device=self.age.device) if env_ids is None
               else torch.as_tensor(env_ids, device=self.age.device, dtype=torch.long))
        if not len(ids):
            return
        chosen = self.rng.integers(len(self.bank.records), size=len(ids))
        self.ids[ids.cpu().numpy()] = chosen
        np.add.at(self.counts, chosen, 1)
        rows = [self.bank.records[int(i)] for i in chosen]
        write_grasp_states(self.adapter, ids, rows, preserve_goal=True)
        target = self.hold.new_tensor([r['hold_target_encoder'] for r in rows])
        limits = self.adapter.limits[ids]
        normalized = 2 * (target - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1
        if (normalized.abs() > 1 + 1e-5).any():
            raise ValueError('Certified warmup target outside policy limits')
        self.hold[ids] = normalized.clamp(-1, 1)
        self.offset[ids] = 0
        self.age[ids] = 0

    @torch.no_grad()
    def action(self):
        refresh = (self.age >= self.settle_steps) & ((self.age - self.settle_steps) % self.refresh_steps == 0)
        noise = (2 * torch.rand(self.offset.shape, generator=self.generator, device=self.offset.device) - 1) * self.exploration
        self.offset = torch.where(refresh[:, None], noise, self.offset)
        # This is the bounded warmup behavior policy, not a changed Actor filter.
        return (self.hold + self.offset).clamp(-1, 1)

    def post_step(self, *_):
        self.age.add_(1)

    def transition_ids(self):
        # Preserve the regular replay schema when temporary warmup is detached.
        return {}

    def detach(self):
        if getattr(self.adapter.env, 'grasp_reset_manager', None) is not self:
            raise ValueError('Warmup reset manager ownership changed')
        del self.adapter.env.grasp_reset_manager

    def metadata(self):
        return dict(bank_path=self.bank.path, bank_sha256=self.bank.sha256, split='train',
                    record_ids=[r['id'] for r in self.bank.records], reset_counts=self.counts.tolist(),
                    behavior='certified hold target + bounded piecewise-constant uniform action offsets',
                    exploration=self.exploration, settle_steps=self.settle_steps, refresh_steps=self.refresh_steps,
                    goals='ordinary task goals preserved', actor_updated=False, transitions='real simulator only')


def readiness_decision(result, position_rmse_m=.02, orientation_rmse_rad=.35, minimum_valid_fraction=.9):
    """Explicit initial readiness budget, not a task-success certificate."""
    if (not all(math.isfinite(v) for v in (position_rmse_m, orientation_rmse_rad, minimum_valid_fraction))
            or position_rmse_m <= 0 or orientation_rmse_rad <= 0 or not 0 < minimum_valid_fraction <= 1):
        raise ValueError('Invalid WM readiness thresholds')
    entry = result['metrics']['8']
    n = len(result.get('environment_ids', [])) or result.get('environments', 128)
    valid = entry.get('valid_fraction')
    if 'valid_fraction' not in entry and n > 0 and entry.get('valid_environments') is not None:
        valid = entry['valid_environments'] / n
    checks = dict(position=entry.get('object_position_vector_rmse_m'),
                  orientation=entry.get('object_orientation_geodesic_rmse_rad'),
                  valid_fraction=valid)
    finite = all(v is not None and math.isfinite(v) for v in checks.values())
    passed = (finite and 0 <= checks['position'] <= position_rmse_m
              and 0 <= checks['orientation'] <= orientation_rmse_rad
              and minimum_valid_fraction <= checks['valid_fraction'] <= 1)
    return dict(passed=bool(passed), observed=checks,
                thresholds=dict(position_rmse_m=position_rmse_m, orientation_rmse_rad=orientation_rmse_rad,
                                minimum_valid_fraction=minimum_valid_fraction),
                scope='8-step prediction readiness for policy updates, not reorientation success')
