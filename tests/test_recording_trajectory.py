"""Recorded terminal transitions and displayed reset states stay distinguishable."""

import numpy as np
import torch

from viserdex.supermimic.recording_trajectory import trajectory_contract, trajectory_snapshot


def test_terminal_snapshot_retains_old_goal_and_copies_live_buffers(tmp_path):
    before = torch.zeros(1, 84)
    before[:, 79:83] = torch.tensor([0., 0., 0., 1.])
    terminal = torch.full((1, 55), 2.)
    displayed = torch.full((1, 84), 3.)
    applied, raw = torch.full((1, 21), .2), torch.full((1, 21), .3)
    visible_object = torch.tensor([[0., 1., 0., 0.]])
    visible_goal = torch.tensor([[1., 0., 0., 0.]])
    base = torch.tensor([[0., 0., 1., 0.]])
    done, terminated = torch.tensor([True]), torch.tensor([False])
    row = dict(state=before, next_physical=terminal, action=applied, done=done, terminated=terminated)
    snapshot = trajectory_snapshot(row, raw, displayed, visible_object, visible_goal, base)
    assert set(snapshot) == set(trajectory_contract()['fields'])
    # Reusing CPU simulator buffers must not rewrite already recorded frames.
    for value in (*row.values(), raw, displayed, visible_object, visible_goal, base):
        value.zero_()
    np.testing.assert_array_equal(snapshot['physical'], np.full(55, 2.))
    np.testing.assert_array_equal(snapshot['post_step_physical'], np.full(55, 3.))
    np.testing.assert_array_equal(snapshot['transition_goal_quaternion_xyzw_base'], [0., 0., 0., 1.])
    np.testing.assert_array_equal(snapshot['target_quaternions_xyzw'], [1., 0., 0., 0.])
    np.testing.assert_array_equal(snapshot['current_quaternions_xyzw'], [0., 1., 0., 0.])
    np.testing.assert_array_equal(snapshot['base_quaternion_xyzw_world'], [0., 0., 1., 0.])
    np.testing.assert_array_equal(snapshot['actions'], np.full(21, .2, dtype=np.float32))
    np.testing.assert_array_equal(snapshot['raw_actions'], np.full(21, .3, dtype=np.float32))
    assert snapshot['done'].item() is True and snapshot['terminated'].item() is False
    path = tmp_path/'trajectory.npz'
    np.savez_compressed(path, **{key: [value, value] for key, value in snapshot.items()})
    with np.load(path) as saved:
        assert saved['done'].shape == (2,) and saved['done'].dtype == np.bool_
        assert saved['post_step_physical'].shape == (2, 55)
        assert saved['transition_goal_quaternion_xyzw_base'].shape == (2, 4)
