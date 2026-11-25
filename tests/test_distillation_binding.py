"""Teacher provenance and memory-bounded updates for the 4096-environment run."""

import copy
from dataclasses import asdict
import hashlib
import json

import pytest
import torch
import yaml

from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.distillation.config import DistillationCfg
from viserdex.distillation.environment import EnvironmentLoader, contract_sha256
from viserdex.distillation.teacher import teacher_metadata
from viserdex.distillation.trainer import DAggerTrainer
from test_distillation import make_batch, small_model


def test_environment_chunks_preserve_full_batch_update_and_recurrent_resets():
    torch.manual_seed(21)
    torch.set_num_threads(1)
    first = small_model().double()
    second = copy.deepcopy(first)
    batch = make_batch(first)
    batch = {name: value.double() if value.is_floating_point() else value for name, value in batch.items()}
    batch["reset"][2, 0] = True
    full = DAggerTrainer(first, DistillationCfg(epochs=2))
    chunks = DAggerTrainer(second, DistillationCfg(epochs=2, update_env_batch_size=2))
    expected, actual = full.update(batch), chunks.update(batch)
    assert actual == pytest.approx(expected, abs=1e-10, rel=1e-8)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=1e-10, rtol=1e-8)
    assert chunks.iteration == full.iteration == 1
    assert chunks.total_steps == full.total_steps == 15
    assert second.prop_norm.count == 15
    assert all(state["step"] == 2 for state in chunks.optimizer.state.values())


@pytest.mark.parametrize("size", [0, -1])
def test_reject_invalid_environment_chunk_size(size):
    with pytest.raises(ValueError):
        DistillationCfg(update_env_batch_size=size)


def test_safe_environment_loader_reads_only_inert_serialized_values():
    data = yaml.load("tuple: !!python/tuple [1, 2]\nslice: !!python/object/apply:builtins.slice [null, null, null]", Loader=EnvironmentLoader)
    assert data == {"tuple": (1, 2), "slice": slice(None)}
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load("!!python/object/apply:os.system ['touch /tmp/never-execute-yaml']", Loader=EnvironmentLoader)


def _environment():
    return {
        "sim": {"dt": 1 / 120, "device": "cuda:0", "physics": {"solver_cfg": {"nconmax": 1152, "njmax": 512}}},
        "scene": {"num_envs": 4096, "robot": {"spawn": {"usd_path": "compact.usda"}},
                  "object": {"spawn": {"usd_path": "cube.usda"}}},
        "observations": {"policy": {"enable_corruption": True}, "privileged": {"enable_corruption": True}},
        "commands": {"object_pose": {"debug_vis": True, "orientation_success_threshold": 0.1}},
        "decimation": 4,
    }


def test_environment_contract_allows_operational_changes_but_rejects_dynamics_drift():
    original = _environment()
    value = copy.deepcopy(original)
    value["scene"]["num_envs"] = 4
    value["sim"]["device"] = "cuda:1"
    value["sim"]["physics"]["solver_cfg"]["nconmax"] = 2048
    value["scene"]["robot"]["spawn"]["spawn_path"] = "/World/envs/env_0/Robot"
    assert contract_sha256(value) == contract_sha256(original)
    for key, changed in (("decimation", 2),):
        bad = copy.deepcopy(value)
        bad[key] = changed
        assert contract_sha256(bad) != contract_sha256(original)
    value["scene"]["robot"]["spawn"]["usd_path"] = "old_hand.usda"
    assert contract_sha256(value) != contract_sha256(original)


def test_compact_teacher_requires_explicit_matching_xyzw_protocol(tmp_path):
    checkpoint = tmp_path / "model_4800.pt"
    checkpoint.write_bytes(b"metadata-only")
    params = tmp_path / "params"
    params.mkdir()
    (params / "agent.yaml").write_text("experiment_name: viserdex_revo3_compact116_cube30hz\nobs_groups:\n  actor: [policy, privileged]\nactor:\n  class_name: MLPModel\n")
    env = params / "env.yaml"
    physics = "isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager"
    env.write_text(f"usd_path: /assets/generated/usd/RSLCube/RSLCube.usda\nrobot: revo3_right_compact_v4.usda\nsim:\n  physics:\n    class_type: {physics}\n")
    with pytest.raises(ValueError, match="XYZW checkpoint namespace"):
        teacher_metadata(checkpoint, "RSLCube")
    protocol = {"quaternion_order": "xyzw", "observation_dims": {"policy": 49, "privileged": 38},
                "joint_names": list(REVO3_JOINT_NAMES), "physics": physics,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "environment_sha256": hashlib.sha256(env.read_bytes()).hexdigest()}
    sidecar = params / "teacher_protocol.json"
    sidecar.write_text(json.dumps(protocol))
    assert teacher_metadata(checkpoint, "RSLCube")["explicit_protocol"] == protocol
    for key, wrong in (("quaternion_order", "wxyz"), ("checkpoint_sha256", "changed"),
                       ("environment_sha256", "changed"), ("physics", "wrong")):
        sidecar.write_text(json.dumps({**protocol, key: wrong}))
        with pytest.raises(ValueError):
            teacher_metadata(checkpoint, "RSLCube")


def test_legacy_training_config_without_chunk_setting_keeps_resume_semantics():
    cfg = DistillationCfg(epochs=2)
    previous = asdict(cfg)
    previous.pop("update_env_batch_size")
    assert asdict(DistillationCfg(**previous)) == asdict(cfg)
