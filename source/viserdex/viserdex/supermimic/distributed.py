"""Synchronous SuperMimic updates, with one official simulation per torchrun rank.

Gradients are reduced once after each accumulated logical batch and BEFORE
clipping/Adam. Explicit reduction also covers the actor's frozen-WM rollout,
where wrapping every repeated forward in DDP would introduce unused reducers.
"""

from datetime import timedelta
import hashlib
import os

import torch
import torch.distributed as dist


class Distributed:
    def __init__(self, device="cpu", enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.rank = dist.get_rank() if enabled else 0
        self.world_size = dist.get_world_size() if enabled else 1

    @classmethod
    def initialize(cls, device):
        enabled = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if enabled:
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=torch.device(device))
        return cls(device, enabled)

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def reduce(self, tensor, operation="mean"):
        result = tensor.clone()
        if self.enabled:
            dist.all_reduce(result, op=dist.ReduceOp.MAX if operation == "max" else dist.ReduceOp.SUM)
            if operation == "mean":
                result.div_(self.world_size)
        return result

    def any(self, value):
        return bool(self.reduce(torch.tensor(int(value), device=self.device), "max").item())

    @torch.no_grad()
    def moments(self, n, mean, variance):
        if not self.enabled:
            return n, mean, variance
        # Double precision avoids cancellation in E[x^2] - E[x]^2.
        first = mean.double()
        packed = torch.cat((first.new_tensor([n]), first * n, (variance.double() + first.square()) * n))
        packed = self.reduce(packed, "sum")
        count = packed[0]
        width = mean.numel()
        average = packed[1 : 1 + width] / count
        var = (packed[1 + width :] / count - average.square()).clamp_min(0)
        return count, average.to(mean.dtype), var.to(variance.dtype)

    @torch.no_grad()
    def broadcast_models(self, *models):
        if self.enabled:
            for model in models:
                for value in model.state_dict().values():
                    dist.broadcast(value, src=0)

    @torch.no_grad()
    def gradients(self, model):
        if not self.enabled:
            return
        parameters = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is None for p in parameters):
            raise RuntimeError("Missing gradient in a synchronized optimizer update")
        flat = torch.cat([p.grad.reshape(-1) for p in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        for p in parameters:
            p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    def batch_denominator(self, local_count):
        """For unequal valid Critic batches: average(sum loss / (global n / ranks))."""
        total = self.reduce(torch.tensor(float(local_count), device=self.device), "sum")
        if total <= 0:
            raise ValueError("Empty global training batch")
        return total / self.world_size

    @torch.no_grad()
    def assert_models_equal(self, *models):
        digest = hashlib.sha256()
        for model in models:
            for name, value in model.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        value = digest.hexdigest()
        if self.enabled:
            values = [None] * self.world_size
            dist.all_gather_object(values, value)
            if len(set(values)) != 1:
                raise AssertionError(f"Model parameters or normalization buffers diverged: {values}")
        return value

    def metrics(self, local):
        if not self.enabled:
            return dict(local)
        # Counts sum across environments; losses/reward means average across ranks.
        sums = {"terminated", "timeouts", "successes", "drops", "completed_episode_goals", "resets",
                "task_samples", "task_angle_sum_rad", "task_position_sum_m", "task_hand_center_distance_sum_m",
                "task_stable_samples", "post_evaluation_samples", "object_hold_cost_gate_sum"}
        ratios = {'task_mean_angle_rad': 'task_angle_sum_rad',
                  'task_mean_position_m': 'task_position_sum_m',
                  'task_mean_hand_center_distance_m': 'task_hand_center_distance_sum_m',
                  'task_stable_fraction': 'task_stable_samples',
                  'object_hold_cost_gate_mean': 'object_hold_cost_gate_sum'}
        maxima = {"update_seconds", "elapsed_seconds", "gpu_peak_gib"}
        invariant = {"iteration", "real_steps", "replay_allocated_gib", "learning_rate"}
        result = dict(local)
        for key, value in local.items():
            if key == "goals_per_completed_episode" or key in ratios:
                continue
            if key == "parity_max_errors":
                names = (
                    "reward",
                    "ema",
                    "action_second_diff",
                    "hand_pose",
                    "clean_student_observation",
                    "teacher87_reconstruction",
                    "orientation_frame",
                    "position_frame",
                    "direct_observation",
                    "hand_convex_vertices_m",
                    "success_boundary_disagreements",
                    "terminal_reset_state_separation",
                    "fk_position_m",
                    "fk_linear_velocity_m_s",
                    "fk_angular_velocity_rad_s",
                    "fk_quaternion",
                    "object_collision_scale_m",
                    "sphere_inertia_kg_m2",
                    "convex_vertices_m",
                    "convex_inertia_kg_m2",
                    "hold_counter",
                )
                result[key] = {
                    name: float(
                        self.reduce(
                            torch.tensor(value.get(name, 0), device=self.device, dtype=torch.float64),
                            "sum" if name == "success_boundary_disagreements" else "max",
                        )
                    )
                    for name in names
                }
            elif isinstance(value, (float, int)) and key not in invariant:
                # Collection cost totals count transitions across all ranks;
                # their worst threshold ratio must retain the worst rank.
                cost_total = key.startswith('cost_') and key.endswith(('_sum', '_violations'))
                cost_maximum = key.startswith('cost_') and key.endswith('_max_ratio')
                op = "sum" if key in sums or cost_total else ("max" if key in maxima or cost_maximum else "mean")
                number = self.reduce(torch.tensor(value, device=self.device, dtype=torch.float64), op).item()
                result[key] = int(number) if isinstance(value, int) else number
        ended = result.get("terminated", 0) + result.get("timeouts", 0)
        result["goals_per_completed_episode"] = result.get("completed_episode_goals", 0) / ended if ended else None
        if 'task_samples' in result:
            samples = result['task_samples']
            for key, numerator in ratios.items():
                if numerator in result:
                    result[key] = result[numerator] / samples if samples else None
        return result

    def close(self):
        if self.enabled:
            dist.destroy_process_group()
