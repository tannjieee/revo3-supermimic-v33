"""Device-selectable ring buffer; CPU storage remains the default.

Observation history and sensor state are stored as actually observed, not rebuilt
with fresh IID noise. Startup domain parameters are indexed by environment ID.
Terminal physical targets are captured BEFORE auto-reset. Sampling uses the
storage device's RNG: changing storage preserves the sampling distribution, not
the exact cross-device sequence of sampled indices.
"""

import torch


class Replay:
    def __init__(self, capacity, num_envs, device, event_capacity=4096, event_fraction=0.25, event_window=8, adaptive=False, uniform_fraction=.2, *, storage_device="cpu"):
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        if capacity < 1 or num_envs < 1 or not 0 < uniform_fraction <= 1:
            raise ValueError("Invalid replay capacity/uniform floor")
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type not in ("cpu", "cuda"):
            raise ValueError("Replay storage_device must be CPU or CUDA")
        self.adaptive = adaptive
        self.uniform_fraction = uniform_fraction
        self.priority = torch.ones(capacity, num_envs, device=self.storage_device)
        self.storage_device = self.priority.device  # Resolve an implicit CUDA index.
        self.generation = torch.zeros(capacity, dtype=torch.long, device=self.storage_device)
        self.serial = 0
        self.max_priority = (torch.ones((), dtype=torch.float64, device=self.storage_device)
                             if self.storage_device.type == "cuda" else 1.0)
        # Fixed-size reduction avoids CUDA unique/nonzero synchronization. The
        # extra entry absorbs stale generation tokens without modifying replay.
        self._priority_scratch = None
        if adaptive and self.storage_device.type == "cuda":
            self._priority_scratch = torch.zeros(capacity * num_envs + 1, dtype=torch.float64,
                                                 device=self.storage_device)
        self.event_capacity = 0 if adaptive else event_capacity
        self.event_fraction = event_fraction
        self.event_window = event_window
        self.archive = {}
        self.archive_size = self.archive_cursor = 0
        self.data = {}
        self.cursor = 0
        self.size = 0

    def add(self, transition):
        if not self.data:
            self.data = {
                k: torch.empty((self.capacity, *v.shape), dtype=v.dtype, device=self.storage_device) for k, v in transition.items()
            }
        if set(transition) != set(self.data):
            raise ValueError("Replay schema changed")
        for k, v in transition.items():
            if v.shape != self.data[k].shape[1:]:
                raise ValueError(f"Replay shape changed: {k}")
            self.data[k][self.cursor].copy_(v.detach().to(self.storage_device))
        self.serial += 1
        self.generation[self.cursor] = self.serial
        self.priority[self.cursor] = self.max_priority
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._archive_events()

    def sample(self, batch, window=1, *, include_events=True, prioritized=True):
        if self.size < window:
            raise ValueError("Insufficient replay")
        if self.adaptive and window > 1 and prioritized:
            return self._sample_prioritized(batch, window)
        logical = torch.randint(self.size - window + 1, (batch,), device=self.storage_device)
        oldest = self.cursor if self.size == self.capacity else 0
        time = (logical[:, None] + oldest + torch.arange(window, device=self.storage_device)[None]) % self.capacity
        env = torch.randint(self.num_envs, (batch,), device=self.storage_device)
        if window == 1:
            out = {k: v[time[:, 0], env].to(self.device) for k, v in self.data.items()}
        else:
            # Real WM/TD windows need no ten-frame policy histories.
            out = {
                k: self.data[k][time, env[:, None]].to(self.device)
                for k in self._wm_keys()
            }
            done = out["done"]
            out["mask"] = torch.cat((torch.ones(batch, 1, device=self.device), (~done[:, :-1]).float().cumprod(1)), 1)
        out["env_id"] = env.to(self.device)
        if include_events and window == self.event_window and self.archive_size:
            n = min(batch, int(batch * self.event_fraction))
            indices = torch.randint(self.archive_size, (n,), device=self.storage_device)
            for key, values in self.archive.items():
                out[key][:n] = values[indices].to(self.device)
            out["mask"][:n] = 1.0
        
        return out

    def _sample_prioritized(self, batch, window):
        if window != self.event_window:
            raise ValueError("Priorities are bound to the configured WM window")
        oldest = self.cursor if self.size == self.capacity else 0
        starts = (torch.arange(self.size-window+1, device=self.storage_device) + oldest) % self.capacity
        priorities = self.priority[starts].flatten().pow(.6)
        count = len(priorities)
        probability = self.uniform_fraction/count + (1-self.uniform_fraction)*priorities/priorities.sum()
        choice = torch.multinomial(probability, batch, replacement=True)
        slot = starts[choice // self.num_envs]
        env = choice % self.num_envs
        times = (slot[:, None] + torch.arange(window, device=self.storage_device)[None]) % self.capacity
        out = {k: self.data[k][times, env[:, None]].to(self.device)
               for k in self._wm_keys()}
        out["mask"] = torch.cat((torch.ones(batch, 1, device=self.device),
            (~out["done"][:, :-1]).float().cumprod(1)), 1)
        out.update(env_id=env.to(self.device), priority_slot=slot.to(self.device),
                   priority_generation=self.generation[slot].to(self.device),
                   sample_probability=probability[choice].to(self.device),
                   importance_weight=(1/(count*probability[choice])).to(self.device))
        return out

    def _wm_keys(self):
        # Keep old synthetic/legacy buffers loadable. v28 Trainer requires both
        # real reward and terminated and never substitutes done for terminated.
        return ('state', 'next_physical', 'action', 'done') + tuple(
            k for k in ('reward', 'terminated', 'contact') if k in self.data)

    def update_priorities(self, batch, errors):
        """Apply real sequence errors, returning the unique valid update count.

        CUDA storage returns a scalar CUDA tensor rather than synchronizing just
        to materialize the diagnostic count; CPU storage retains its int return.
        Finite/nonnegative validation remains immediate on both storage devices.
        """
        if not self.adaptive or "priority_slot" not in batch:
            return 0
        values = errors.detach().to(self.storage_device).flatten()
        if not (torch.isfinite(values) & (values >= 0)).all():
            raise ValueError("Priorities require finite nonnegative real-sequence errors")
        slots, envs, generations = (batch[k].detach().to(self.storage_device) for k in
            ("priority_slot", "env_id", "priority_generation"))
        if any(values.shape != token.shape for token in (slots, envs, generations)):
            raise ValueError("One priority per sampled sequence is required")
        if self.storage_device.type == "cuda":
            return self._update_priorities_resident(values, slots, envs, generations)
        valid = self.generation[slots] == generations
        # Reduce repeated draws with max, making the update order independent.
        indices = slots[valid] * self.num_envs + envs[valid]
        unique, inverse = torch.unique(indices, return_inverse=True)
        # Match the former Python-float epsilon/clamp before storing float32.
        updates = (values[valid].double() + 1e-5).clamp_max(1e6)
        maxima = torch.zeros(len(unique), dtype=updates.dtype)
        maxima.scatter_reduce_(0, inverse, updates, reduce="amax", include_self=True)
        self.priority.view(-1)[unique] = maxima.to(self.priority.dtype)
        if len(unique):
            self.max_priority = max(self.max_priority, float(maxima.max()))
        return len(unique)

    def _update_priorities_resident(self, values, slots, envs, generations):
        """Fixed-size device reduction; inputs are already checked and resident."""
        # Also permits switching adaptive on after construction, as CPU callers
        # historically could. Allocate once, never in the steady-state path.
        if self._priority_scratch is None:
            self._priority_scratch = torch.zeros(self.capacity * self.num_envs + 1,
                                                 dtype=torch.float64, device=self.storage_device)
        valid = self.generation[slots] == generations
        sentinel = self.priority.numel()
        indices = torch.where(valid, slots * self.num_envs + envs, sentinel)
        updates = torch.where(valid, (values.double() + 1e-5).clamp_max(1e6), 0.0)
        self._priority_scratch.zero_()
        self._priority_scratch.scatter_reduce_(0, indices, updates, reduce="amax", include_self=True)
        maxima = self._priority_scratch[:-1]
        touched = maxima > 0  # Every valid update is at least the 1e-5 floor.
        flat = self.priority.view(-1)
        flat.copy_(torch.where(touched, maxima.to(flat.dtype), flat))
        # Retain the old Python-float precision before writing float32 priority.
        self.max_priority = torch.maximum(torch.as_tensor(self.max_priority, dtype=torch.float64,
                                                          device=self.storage_device), maxima.max())
        return touched.sum()

    def sample_real(self, batch):
        """Uniform transitions, without actor histories or event oversampling."""
        if not self.size:
            raise ValueError("Insufficient replay")
        time = torch.randint(self.size, (batch,), device=self.storage_device)
        env = torch.randint(self.num_envs, (batch,), device=self.storage_device)
        keys = ("state", "next_physical", "next_aux", "reward", "terminated")
        # Legacy synthetic test fixtures may store the complete next state.
        if "next_aux" not in self.data:
            keys = ("state", "next_state", "reward", "terminated")
        out = {k: self.data[k][time, env].to(self.device) for k in keys}
        if "next_aux" in out:
            out["next_state"] = torch.cat((out.pop("next_physical"), out.pop("next_aux")), -1)
        out["env_id"] = env.to(self.device)
        return out

    def _archive_events(self):
        # Compact windows only; no actor history or simulated sensor copies.
        w = self.event_window
        if not self.event_capacity or self.size < w:
            return
        times = (torch.arange(w, device=self.storage_device) + self.cursor - w) % self.capacity
        state = self.data["state"][times]
        nxt = self.data["next_physical"][times]
        done = self.data["done"][times]
        distance = nxt[-1, :, 42:45].norm(dim=-1)
        speed = nxt[-1, :, 49:52].norm(dim=-1)
        from .state_layout import layout
        from .core import ROT, GOAL_ROT, quat_mul, quat_conjugate, rotation_log
        angle = rotation_log(quat_mul(nxt[-1, :, ROT], quat_conjugate(state[-1, :, layout(state).goal_rot]))).norm(dim=-1)
        previous = state[-1, :, 42:45].norm(dim=-1)
        # Motion proxies, explicitly NOT measured contact/slip labels.
        event = done[-1] | (distance > .20) | (speed > 1.) | (angle < .1)
        event |= (previous > .15) & (distance < previous - .02)
        valid = event & ~done[:-1].any(0)
        ids = valid.nonzero().flatten()
        if not len(ids):
            return
        ids = ids[torch.randperm(len(ids), device=self.storage_device)[:16]]
        windows = {k: self.data[k][times[:, None], ids[None]].transpose(0, 1).clone()
                   for k in self._wm_keys()}
        windows["env_id"] = ids
        if not self.archive:
            self.archive = {k: torch.empty((self.event_capacity, *v.shape[1:]), dtype=v.dtype,
                                           device=self.storage_device) for k,v in windows.items()}
        for j in range(len(ids)):
            for k,v in windows.items():
                self.archive[k][self.archive_cursor].copy_(v[j])
            self.archive_cursor = (self.archive_cursor + 1) % self.event_capacity
            self.archive_size = min(self.archive_size + 1, self.event_capacity)

    @property
    def allocated_bytes(self):
        tensors = [*self.data.values(), *self.archive.values(), self.priority, self.generation]
        if self._priority_scratch is not None:
            tensors.append(self._priority_scratch)
        if isinstance(self.max_priority, torch.Tensor):
            tensors.append(self.max_priority)
        return sum(t.numel() * t.element_size() for t in tensors)
