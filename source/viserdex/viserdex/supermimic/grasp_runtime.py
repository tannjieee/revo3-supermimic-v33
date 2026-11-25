"""Certified reset states through official recorder hooks; no action override."""
import numpy as np
import torch
from .core import quat_apply, quat_mul, quat_conjugate
from .grasp_bank import PairSampler


@torch.no_grad()
def write_grasp_states(adapter, env_ids, records, goals=None, velocity=None, preserve_goal=False):
    """Call only after the official reset cleared actuator/action memory."""
    from .simulation import tensor
    ids = torch.as_tensor(env_ids, device=adapter.env.device, dtype=torch.long)
    q = adapter.state.new_tensor([r['q_geometry'] for r in records])
    pose = adapter.state.new_tensor([r['object_pose_xyzw'] for r in records])
    base = quat_conjugate(adapter.base_inverse[ids])
    world = torch.cat((tensor(adapter.robot.data.root_pos_w)[ids] + quat_apply(base, pose[:, :3]),
                       quat_mul(base, pose[:, 3:])), -1)
    v = torch.zeros(len(ids), 6, device=ids.device) if velocity is None else pose.new_tensor(velocity).expand(len(ids), -1)
    world_v = torch.cat((quat_apply(base, v[:, :3]), quat_apply(base, v[:, 3:])), -1)
    adapter.robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q),
                                                 joint_ids=adapter.ids, env_ids=ids)
    adapter.obj.write_root_pose_to_sim_index(root_pose=world, env_ids=ids)
    adapter.obj.write_root_velocity_to_sim_index(root_velocity=world_v, env_ids=ids)
    # No hold target, previous action or delayed command is installed here.
    if not preserve_goal:
        goal = pose if goals is None else pose.new_tensor([r['object_pose_xyzw'] for r in goals])
        adapter.env.command_manager.get_term('object_pose').set_goal(ids, goal[:, 3:], goal[:, :3])


@torch.no_grad()
def synchronize_reset(adapter):
    """Rebuild the first actual observation after an explicit full env.reset()."""
    from .sensors import reset_sensors, observe
    from .context import audit_delay_memory
    adapter.raw_history.zero_()
    adapter.delay_ready.zero_()
    adapter.sensors = reset_sensors(adapter.sensors, torch.ones(adapter.env.num_envs, device=adapter.env.device,
                                                               dtype=torch.bool), adapter.noise_cfg)
    adapter.state = adapter.read_state()
    obs, adapter.sensors = observe(adapter.state, adapter.sensors, adapter.limits, adapter.noise_cfg, adapter.hold_cfg)
    adapter.history = obs[:, None].expand_as(adapter.history).clone()
    audit_delay_memory(adapter.delay_ready)
    adapter.audit_geometry(adapter.state)


class GraspResetManager:
    def __init__(self, adapter, bank, seed=0, adaptive=True, evaluate=False):
        self.adapter, self.bank = adapter, bank
        self.sampler = PairSampler(bank.records, seed, adaptive and not evaluate)
        self.evaluate = evaluate
        n = adapter.env.num_envs
        self.starts = np.zeros(n, dtype=int)
        self.goals = np.zeros(n, dtype=int)
        self.buckets = np.zeros(n, dtype=int)
        self.succeeded = np.zeros(n, dtype=bool)
        self.episodes = 0
        adapter.env.grasp_reset_manager = self

    def reset(self, env_ids):
        ids = np.arange(self.adapter.env.num_envs) if env_ids is None else np.asarray(
            env_ids.cpu() if isinstance(env_ids, torch.Tensor) else env_ids)
        starts, goals, buckets = self.sampler.sample(len(ids))
        self.starts[ids], self.goals[ids], self.buckets[ids] = starts, goals, buckets
        self.succeeded[ids] = False
        write_grasp_states(self.adapter, ids, [self.bank.records[i] for i in starts],
                           [self.bank.records[i] for i in goals])

    def post_step(self, done, success):
        self.succeeded |= success.detach().cpu().numpy()
        ids = done.nonzero().flatten().cpu().numpy()
        if len(ids):
            if not self.evaluate:
                self.sampler.update(self.buckets[ids], self.succeeded[ids])
            self.episodes += len(ids)

    def transition_ids(self):
        return {name: torch.tensor(value.copy(), device=self.adapter.env.device) for name, value in
                (('grasp_start_id', self.starts), ('grasp_goal_id', self.goals), ('grasp_bucket', self.buckets))}

    def audit_transition(self, transition, reset_state):
        """Assert old-episode labels and actual new-episode reset data stay separate."""
        state = transition['state']
        goals = state.new_tensor([self.bank.records[int(i)]['object_pose_xyzw']
                                  for i in transition['grasp_goal_id']])
        torch.testing.assert_close(state[:, 76:79], goals[:, :3], atol=2e-6, rtol=0)
        assert torch.allclose((state[:, 79:83]*goals[:, 3:]).sum(-1).abs(),
                              torch.ones(len(state), device=state.device), atol=2e-6, rtol=0)
        ids = transition['done'].nonzero().flatten()
        if len(ids):
            records = [self.bank.records[self.starts[int(i)]] for i in ids]
            expected_q = state.new_tensor([r['q_geometry'] for r in records])
            expected_pose = state.new_tensor([r['object_pose_xyzw'] for r in records])
            torch.testing.assert_close(reset_state[ids, :21], expected_q, atol=2e-6, rtol=0)
            torch.testing.assert_close(reset_state[ids, 42:45], expected_pose[:, :3], atol=2e-6, rtol=0)
            torch.testing.assert_close(reset_state[ids, 55:76], torch.zeros_like(expected_q), atol=0, rtol=0)
            assert not reset_state[ids, 83].any()

    def state_dict(self):
        return dict(bank_sha256=self.bank.sha256, split=self.bank.split, sampler=self.sampler.state_dict(),
                    episodes=self.episodes, statistics_scope='single_process')

    def load_state_dict(self, state):
        if state['bank_sha256'] != self.bank.sha256:
            raise ValueError('Resume grasp bank content changed')
        # Evaluation may use another split, but must not inherit train difficulty/RNG.
        if not self.evaluate:
            if state['split'] != self.bank.split:
                raise ValueError('Resume grasp split changed')
            self.sampler.load_state_dict(state['sampler'])
            self.episodes = int(state['episodes'])
