"""Belief student with a deployment interface that cannot accept privileged state."""

import torch
from torch import nn

from .config import StudentCfg


def mlp(input_dim, hidden, output_dim):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class RunningNormalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.var.copy_((self.var * self.count + var * n + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def forward(self, x):
        # Unit scale before the first rollout, lower bound for near-constant features.
        return ((x - self.mean) / self.var.clamp_min(1e-4).sqrt()).clamp(-10, 10)


class BeliefStudent(nn.Module):
    def __init__(self, cfg: StudentCfg):
        super().__init__()
        self.cfg = cfg
        self.prop_norm = RunningNormalizer(cfg.proprio_dim)
        self.extero_norm = RunningNormalizer(cfg.extero_dim)
        self.target_norm = RunningNormalizer(cfg.extero_dim + cfg.privileged_dim)
        self.extero_encoder = mlp(cfg.extero_dim, cfg.extero_hidden, cfg.extero_latent)
        self.rnn = nn.GRU(cfg.proprio_dim + cfg.extero_latent, cfg.rnn_hidden, cfg.rnn_layers)
        self.belief = mlp(cfg.rnn_hidden, cfg.latent_hidden, cfg.belief_dim)
        self.gate = mlp(cfg.rnn_hidden, cfg.gate_hidden, cfg.extero_latent)
        self.extero_memory = mlp(cfg.rnn_hidden, cfg.latent_hidden, cfg.extero_latent)
        self.actor = mlp(cfg.proprio_dim + cfg.extero_latent + cfg.belief_dim, cfg.actor_hidden, cfg.num_actions)
        self.extero_decoder = mlp(cfg.belief_dim + cfg.extero_dim, cfg.decoder_hidden, cfg.extero_dim)
        self.privileged_decoder = mlp(cfg.belief_dim, cfg.decoder_hidden, cfg.privileged_dim)

    def initial_state(self, batch_size, device):
        return torch.zeros(self.cfg.rnn_layers, batch_size, self.cfg.rnn_hidden, device=device)

    def forward(self, proprio, extero, hidden):
        prop = self.prop_norm(proprio)
        ext = self.extero_norm(extero)
        encoded = self.extero_encoder(ext)
        rnn_out, hidden = self.rnn(torch.cat((prop, encoded), -1).unsqueeze(0), hidden)
        memory = rnn_out.squeeze(0)
        belief = self.belief(memory)
        fused = torch.sigmoid(self.gate(memory)) * encoded + self.extero_memory(memory)
        action = self.actor(torch.cat((prop, fused, belief), -1))
        reconstruction = torch.cat(
            (self.extero_decoder(torch.cat((belief, ext), -1)), self.privileged_decoder(belief)), -1
        )
        return action, reconstruction, hidden

    def sequence(self, proprio, extero, hidden, reset_before):
        # Feed-forward layers have no temporal dependencies. Batch their work
        # over time/environment rather than launching the full model per step.
        prop = self.prop_norm(proprio)
        ext = self.extero_norm(extero)
        encoded = self.extero_encoder(ext)
        inputs = torch.cat((prop, encoded), -1)
        reset_steps = reset_before.any(dim=1).nonzero(as_tuple=False).flatten().cpu().tolist()
        boundaries = [0, *[step for step in reset_steps if step > 0], proprio.shape[0]]
        memories = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            # Split only at actual reset boundaries. Other environments retain
            # their hidden state; gradients cannot cross a reset for that env.
            hidden = hidden * (~reset_before[start]).to(hidden.dtype)[None, :, None]
            memory, hidden = self.rnn(inputs[start:stop], hidden)
            memories.append(memory)
        memory = torch.cat(memories, dim=0)
        belief = self.belief(memory)
        fused = torch.sigmoid(self.gate(memory)) * encoded + self.extero_memory(memory)
        action = self.actor(torch.cat((prop, fused, belief), -1))
        reconstruction = torch.cat(
            (self.extero_decoder(torch.cat((belief, ext), -1)), self.privileged_decoder(belief)), -1
        )
        return action, reconstruction, hidden


class DeploymentPolicy(nn.Module):
    """Explicit hidden-state input/output; caller clears it at episode boundaries."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, proprio, estimated_extero, hidden):
        action, _, hidden = self.student(proprio, estimated_extero, hidden)
        return action, hidden
