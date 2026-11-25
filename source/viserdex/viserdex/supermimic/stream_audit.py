"""Frozen, future-data WM audits with rotating environment coverage."""
import json
from pathlib import Path
import torch
from .core import physical_delta
from .networks import Dynamics
from .validation import open_loop_errors


@torch.no_grad()
def audit_rollout(model, records, context, limits, cfg):
    """Open-loop errors plus per-environment tails, without a second model run.

    Callers must freeze the model before collecting these records and evaluate
    before fitting them. No acceptance threshold is imposed here. Invalidated
    post-reset predictions are masked exactly as in ``open_loop_errors``.
    """
    if not records:
        raise ValueError("An audit needs at least one real transition")
    predictions = []

    class Recorder:
        geometry = getattr(model, "geometry", None)

        def __call__(self, state, target, context):
            result = model(state, target, context)
            predictions.append(result.detach())
            return result

    errors = open_loop_errors(Recorder(), records, context, limits, cfg)
    initial = records[0]["state"]
    mask = torch.ones(len(initial), device=initial.device, dtype=torch.bool)
    for step, (row, prediction) in enumerate(zip(records, predictions), 1):
        if step in (1, 8, 16, 32):
            entry = errors[str(step)]
            if cfg.wm_task_losses:
                from .termination_calibration import binary_metrics
                from .spatial_contract import drop_distance
                logits = model.termination_logits(prediction)[mask]
                labels = row['terminated'][mask]
                entry['termination_raw'] = binary_metrics(logits, labels)
                entry['termination_calibrated'] = binary_metrics(
                    logits / model.termination_temperature + model.termination_bias, labels)
                entry['termination_analytic_baseline'] = binary_metrics(
                    ((drop_distance(prediction, cfg) - cfg.drop_radius_m) / cfg.survival_width)[mask], labels)
            entry["valid_fraction"] = entry["valid_environments"] / len(initial)
            predicted_error = physical_delta(row["next_physical"], prediction)
            persistence_error = physical_delta(row["next_physical"], initial)
            for prefix, delta in (("", predicted_error), ("persistence_", persistence_error)):
                for name, units, sl in (("object_position_vector", "m", slice(42, 45)),
                                        ("object_orientation_geodesic", "rad", slice(45, 48))):
                    norms = delta[mask, sl].norm(dim=-1)
                    # Include unambiguous vector/geodesic persistence RMSE too.
                    entry[f"{prefix}{name}_rmse_{units}"] = float(norms.square().mean().sqrt()) if len(norms) else None
                    tails = torch.quantile(norms, norms.new_tensor([.95, .99])).cpu().tolist() if len(norms) else (None, None)
                    entry[f"{prefix}{name}_p95_{units}"], entry[f"{prefix}{name}_p99_{units}"] = tails
        mask &= ~row["done"]
    return errors


class StreamAudit:
    def __init__(self, path, context, limits, cfg, geometry_spec, interval=200, environments=128):
        if type(environments) is not int or environments < 1 or len(context) < 1:
            raise ValueError("An audit needs positive environment coverage")
        self.path, self.cfg, self.interval = Path(path), cfg, interval
        self.n = min(environments, len(context))
        self.all_context, self.all_limits = context, limits
        self.rows, self.last = [], None
        self.policy_tanh_slopes = []
        self.latest_result = None
        self.started = self.completed = 0
        self.environment_ids = None
        if interval <= 0:
            return
        device = context.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        # Constructing the audit must not consume training/exploration RNG.
        with torch.random.fork_rng(devices=devices):
            self.model = Dynamics(context.shape[-1], cfg, geometry_spec).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def start(self, model, iteration, environment_ids=None):
        """Freeze a new future-data window, including at a fixed readiness step.

        The default selects one world from each contiguous stratum, rotating
        its offset each window. Eight 128-world windows cover all 1024 worlds,
        including world 801 in the second window. No RNG stream is consumed.
        """
        if self.interval <= 0:
            raise ValueError("Cannot start a disabled stream audit")
        if self.rows or self.environment_ids is not None:
            raise ValueError("Finish the current audit before starting another")
        total = len(self.all_context)
        if environment_ids is None:
            starts = torch.arange(self.n, device=self.all_context.device) * total // self.n
            stops = torch.arange(1, self.n + 1, device=self.all_context.device) * total // self.n
            ids = starts + self.started % (stops - starts)
        else:
            ids = torch.as_tensor(environment_ids, device=self.all_context.device)
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not len(ids):
                raise ValueError("Audit environment IDs must be a nonempty integer vector")
            if int(ids.min()) < 0 or int(ids.max()) >= total or len(ids.unique()) != len(ids):
                raise ValueError("Audit environment IDs must be unique and in range")
        self.environment_ids = ids.long()
        self.context, self.limits = self.all_context[ids], self.all_limits[ids]
        self.last = int(iteration)
        self.started += 1
        self.model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def observe(self, model, row, iteration, *, policy_action=None):
        """Return a completed result (otherwise None); first window starts now.

        ``policy_action`` is the actual tanh output before post-tanh exploration.
        External behavior actions have no such source and leave it unavailable.
        Only the per-environment mean slope is cached for this 32-step audit;
        the real Replay schema and executed-action diagnostics stay unchanged.
        """
        if self.interval <= 0:
            return None
        if not self.rows and self.environment_ids is None:
            if self.last is not None and iteration < self.last + self.interval:
                return None
            self.start(model, iteration)
        if policy_action is not None and policy_action.shape != row['action'].shape:
            raise ValueError('Policy tanh diagnostics require the full executed-action shape')
        keys = ('state', 'action', 'next_physical', 'done', 'reward') + (('terminated',) if self.cfg.wm_task_losses else ())
        self.rows.append({k: row[k][self.environment_ids].detach().clone() for k in keys})
        if policy_action is not None:
            selected = policy_action[self.environment_ids].detach()
            self.policy_tanh_slopes.append((1-selected.square()).mean(-1))
        if len(self.rows) < 32:
            return None
        errors = audit_rollout(self.model, self.rows, self.context, self.limits, self.cfg)
        actions = torch.stack([r['action'] for r in self.rows])
        slopes = torch.cat(self.policy_tanh_slopes) if self.policy_tanh_slopes else None
        result = dict(iteration=self.last, audit_index=self.started - 1, control_dt=self.cfg.control_dt,
                      environment_ids=self.environment_ids.cpu().tolist(),
                      metrics={k: errors[k] for k in ('1', '8', '16', '32')},
                      action_saturated_fraction=float((actions.abs() > 1).float().mean()),
                      action_near_bound_fraction=float((actions.abs() > .98).float().mean()),
                      action_tanh_slope_mean=float(slopes.mean()) if slopes is not None else None,
                      action_tanh_slope_valid_samples=slopes.numel() if slopes is not None else 0,
                      action_tanh_slope_total_samples=actions.shape[0]*actions.shape[1])
        with self.path.open('a') as f:
            f.write(json.dumps(result)+'\n')
        self.latest_result = result
        self.completed += 1
        self.rows.clear()
        self.policy_tanh_slopes.clear()
        self.environment_ids = None
        return result
