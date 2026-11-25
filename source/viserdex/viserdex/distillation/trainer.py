"""Online DAgger with whole-environment mixing and reset-aware truncated BPTT."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import DistillationCfg


def mix_actions(teacher, student, beta):
    """Select one policy per environment, not a blend or independent joint choices."""
    selected = torch.rand(teacher.shape[0], 1, device=teacher.device) < beta
    return torch.where(selected, teacher, student), selected


class DAggerTrainer:
    def __init__(self, student, cfg: DistillationCfg):
        self.student = student
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.iteration = 0  # completed updates; beta is never reset on resume
        self.total_steps = 0

    def update(self, batch):
        student, cfg = self.student, self.cfg
        student.train()
        # Once per collected sample, not once per optimization epoch.
        student.prop_norm.update(batch["prop"])
        student.extero_norm.update(batch["extero"])
        student.target_norm.update(batch["target"])
        target = student.target_norm(batch["target"])
        total_metrics = torch.zeros(4, dtype=torch.float64, device=batch["prop"].device)
        num_envs = batch["prop"].shape[1]
        chunk_size = cfg.update_env_batch_size or num_envs
        optimizer_size = min(cfg.optimizer_env_batch_size or num_envs, num_envs)
        optimizer_steps = 0
        for _ in range(cfg.epochs):
            # Shuffle environment sequences, never individual time steps. The
            # memory chunk size does not decide the number of AdamW updates.
            order = torch.randperm(num_envs, device=batch["prop"].device) if optimizer_size < num_envs else None
            for logical_start in range(0, num_envs, optimizer_size):
                logical_stop = min(logical_start + optimizer_size, num_envs)
                logical_count = logical_stop - logical_start
                self.optimizer.zero_grad(set_to_none=True)
                logical_metrics = torch.zeros(3, dtype=torch.float64, device=batch["prop"].device)
                for start in range(logical_start, logical_stop, chunk_size):
                    stop = min(start + chunk_size, logical_stop)
                    ids = slice(start, stop) if order is None else order[start:stop]
                    weight = (stop - start) / logical_count
                    action, recon, _ = student.sequence(
                        batch["prop"][:, ids], batch["extero"][:, ids],
                        batch["hidden"][:, ids], batch["reset"][:, ids],
                    )
                    bc = F.mse_loss(action, batch["teacher"][:, ids])
                    d = student.cfg.extero_dim
                    expected = target[:, ids]
                    extero_loss = F.mse_loss(recon[..., :d], expected[..., :d])
                    extero_loss = extero_loss + cfg.decoder_l1_coef * F.l1_loss(recon[..., :d], expected[..., :d])
                    priv_loss = F.mse_loss(recon[..., d:], expected[..., d:])
                    priv_loss = priv_loss + cfg.decoder_l1_coef * F.l1_loss(recon[..., d:], expected[..., d:])
                    reconstruction = cfg.decoder_extero_coef * extero_loss + priv_loss
                    loss = bc + cfg.reconstruction_coef * reconstruction
                    (loss * weight).backward()
                    logical_metrics.add_(
                        torch.stack((loss.detach(), bc.detach(), reconstruction.detach())).double(), alpha=weight
                    )
                if not torch.isfinite(logical_metrics).all():
                    raise FloatingPointError("Non-finite distillation loss; checkpoint was not advanced")
                grad = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                optimizer_steps += 1
                relative_count = logical_count / num_envs
                total_metrics[:3].add_(logical_metrics, alpha=relative_count)
                total_metrics[3].add_(grad.detach().double(), alpha=relative_count)
        self.iteration += 1
        self.total_steps += batch["prop"].shape[0] * batch["prop"].shape[1]
        student.eval()
        metrics = dict(zip(("loss", "bc", "reconstruction", "grad_norm"), (total_metrics / cfg.epochs).cpu().tolist()))
        metrics.update(optimizer_steps=optimizer_steps, optimizer_batch_envs=optimizer_size)
        return metrics

    @torch.no_grad()
    def replay_hidden(self, batch):
        """Rebuild rollout-end memory using updated weights and normalization."""
        chunk = self.cfg.update_env_batch_size or batch["prop"].shape[1]
        values = []
        for start in range(0, batch["prop"].shape[1], chunk):
            ids = slice(start, start + chunk)
            _, _, hidden = self.student.sequence(
                batch["prop"][:, ids], batch["extero"][:, ids],
                batch["hidden"][:, ids], batch["reset"][:, ids],
            )
            values.append(hidden)
        result = torch.cat(values, dim=1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite rebuilt recurrent state")
        return result

    def save(self, path, metadata, noise_cfg):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "viserdex_revo3_belief_v1",
            "student_cfg": asdict(self.student.cfg),
            "training_cfg": asdict(self.cfg),
            "noise_cfg": asdict(noise_cfg),
            "student": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "metadata": metadata,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, payload):
        if asdict(DistillationCfg(**payload["training_cfg"])) != asdict(self.cfg):
            raise ValueError("Resume must preserve the saved distillation configuration")
        self.student.load_state_dict(payload["student"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = payload["iteration"]
        self.total_steps = payload["total_steps"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])


def load_student_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "viserdex_revo3_belief_v1":
        raise ValueError("Expected a Revo3 Sim 6 belief-student checkpoint, not a PPO or Sim 5 checkpoint")
    return payload
