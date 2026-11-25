"""Compatibility checks for the goal distribution used by the ViserDex release."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from isaaclab_tasks.core.reorient.utils import SuccessTracker
from viserdex.tasks.manipulation.inhand.config.allegro_hand.allegro_env_cfg import ViserDexAllegroEnvCfg
from viserdex.tasks.manipulation.inhand.config.revo3_hand.revo3_env_cfg import ViserDexRevo3EnvCfg
from viserdex.tasks.manipulation.inhand.mdp.commands import ViserDexReorientCommand


@pytest.mark.parametrize("env_ids", [torch.tensor([7, 1, 5]), [6, 2], [], slice(None)])
@pytest.mark.parametrize("make_quat_unique", [False, True])
def test_sampler_matches_released_rotation_composition(env_ids, make_quat_unique):
    command = object.__new__(ViserDexReorientCommand)
    command._env = SimpleNamespace(num_envs=8, device="cpu")
    command.cfg = SimpleNamespace(make_quat_unique=make_quat_unique)
    command._success = SuccessTracker(8, "cpu")
    command.quat_command_w = torch.zeros(8, 4)
    command.quat_command_w[:, 3] = 1.0
    before = command.quat_command_w.clone()
    ids = torch.arange(8)[env_ids]

    # Independent SciPy rotations verify the source's q_x * q_y ordering and
    # v3 XYZW layout, including non-contiguous selective resets.
    generator = torch.Generator().manual_seed(20261020)
    angles = (2.0 * torch.rand(len(ids), 2, generator=generator) - 1.0) * torch.pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261020)
        command._resample_command(env_ids)
        # The original sampler consumes exactly two uniform draws per target.
        torch.testing.assert_close(torch.rand(5), torch.rand(5, generator=generator))

    if len(ids):
        expected = (
            Rotation.from_rotvec(np.column_stack((angles[:, 0].numpy(), np.zeros((len(ids), 2)))))
            * Rotation.from_rotvec(np.column_stack((np.zeros(len(ids)), angles[:, 1].numpy(), np.zeros(len(ids)))))
        ).as_quat()
        if make_quat_unique:
            expected[expected[:, 3] < 0] *= -1
        actual = command.quat_command_w[ids]
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(len(ids)), atol=2e-6, rtol=0)
        # Old targets keep the body's Y axis in the reference YZ plane.
        matrices = Rotation.from_quat(actual.numpy()).as_matrix()
        np.testing.assert_allclose(matrices[:, 0, 1], 0.0, atol=2e-6)

    untouched = torch.ones(8, dtype=torch.bool)
    untouched[ids] = False
    torch.testing.assert_close(command.quat_command_w[untouched], before[untouched])
    expected_successes = torch.zeros(8)
    expected_successes[ids] = 1.0
    torch.testing.assert_close(command._success.snapshot(slice(None)), expected_successes)


@pytest.mark.parametrize("cfg_type", [ViserDexAllegroEnvCfg, ViserDexRevo3EnvCfg])
def test_teacher_task_uses_released_sampler(cfg_type):
    cfg = cfg_type()
    command_cfg = cfg.commands.object_pose
    assert command_cfg.class_type is ViserDexReorientCommand
    assert command_cfg.update_goal_on_success
    assert command_cfg.orientation_success_threshold == 0.1
