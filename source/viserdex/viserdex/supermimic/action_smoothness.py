"""Reset-aware raw-action second difference, evaluated once per reward step."""
import torch
from isaaclab.managers import ManagerTermBase


class ActionSecondDifference(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_previous = torch.zeros_like(env.action_manager.action)
        self.last_penalty = torch.zeros(env.action_manager.action.shape[0], device=env.action_manager.action.device)

    def reset(self, env_ids=None):
        self.previous_previous[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        current = env.action_manager.action
        previous = env.action_manager.prev_action
        penalty = (current - 2 * previous + self.previous_previous).square().sum(-1)
        self.last_penalty.copy_(penalty.detach())
        self.previous_previous.copy_(previous.detach())
        return penalty
