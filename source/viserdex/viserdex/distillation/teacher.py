"""Load a frozen RSL-RL 5 actor together with its saved normalization statistics."""

import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml


TEACHER_OBSERVATION_PROTOCOL = "shared_noisy_joints_clean_privileged_v1"


def teacher_observation_protocol(quaternion_sign="raw"):
    if quaternion_sign == "raw":
        return TEACHER_OBSERVATION_PROTOCOL
    if quaternion_sign == "positive_world_w":
        return "shared_noisy_joints_canonical_world_quaternions_v2"
    raise ValueError("Unsupported teacher quaternion sign convention")


def canonical_teacher_privileged(privileged):
    """Choose one representation for physically identical q/-q teacher inputs."""
    result = privileged.clone()
    sign = torch.where(result[:, 27:28] < 0, -1.0, 1.0)
    result[:, 24:28] *= sign
    # The raw goal difference is object * conjugate(goal), so it has the
    # same sign change. Positions and all velocity targets stay exact.
    result[:, 34:38] *= sign
    return result


def configure_clean_state_observations(env_cfg):
    """Keep the source state clean; inject shared sensor noise after observation assembly."""
    for group in ("policy", "privileged"):
        getattr(env_cfg.observations, group).enable_corruption = False


@torch.no_grad()
def audit_clean_state_observations(env, observations):
    """Compare source state terms to direct, uncorrupted simulator values.

    The observation manager resolves entity/joint selections into its own cfg.
    This deliberately fails on future modifiers/history rather than claiming
    isolation without knowing the observation transformation.
    """
    errors = {}
    for name in ("policy", "privileged"):
        group = getattr(env.observation_manager.cfg, name)
        if group.enable_corruption:
            raise AssertionError(f"Source state {name} observation corruption must be disabled")
        values = []
        for term_name in env.observation_manager.active_terms[name]:
            term = getattr(group, term_name)
            if term.modifiers or term.history_length:
                raise ValueError("Source state audit does not support observation modifiers/history")
            value = term.func(env, **term.params).clone()
            if term.clip is not None:
                value = value.clamp(*term.clip)
            if term.scale is not None:
                value *= torch.as_tensor(term.scale, device=value.device)
            values.append(value)
        expected = torch.cat(values, -1)
        torch.testing.assert_close(observations[name], expected, atol=1e-6, rtol=1e-6)
        errors["clean_state_" + name + "_max_abs_error"] = float((observations[name] - expected).abs().max())
    return errors


def teacher_observations(clean_observations, student_proprio, quaternion_sign="raw"):
    """Give both actors the same encoder sample; keep teacher-only state exact.

    The released teacher's policy layout is joints[21], goal[7], last_action[21].
    Goal/action commands are software state, so only the measured joints are noisy.
    """
    if clean_observations["policy"].shape[-1] != 49 or student_proprio.shape[-1] != 113:
        raise ValueError("Unexpected Revo3 teacher/student proprioceptive layout")
    result = clean_observations.copy()
    result["policy"] = torch.cat((student_proprio[:, :21], clean_observations["policy"][:, 21:]), dim=-1)
    if quaternion_sign == "positive_world_w":
        result["privileged"] = canonical_teacher_privileged(clean_observations["privileged"])
    elif quaternion_sign != "raw":
        raise ValueError("Unsupported teacher quaternion sign convention")
    return result


@torch.no_grad()
def audit_teacher_observations(observations, clean_observations, student_proprio, quaternion_sign="raw"):
    torch.testing.assert_close(observations["policy"][:, :21], student_proprio[:, :21], atol=0, rtol=0)
    torch.testing.assert_close(observations["policy"][:, 21:], clean_observations["policy"][:, 21:], atol=0, rtol=0)
    expected = clean_observations["privileged"]
    if quaternion_sign == "positive_world_w":
        expected = canonical_teacher_privileged(expected)
    elif quaternion_sign != "raw":
        raise ValueError("Unsupported teacher quaternion sign convention")
    torch.testing.assert_close(observations["privileged"], expected, atol=0, rtol=0)
    return {
        "teacher_joint_noise_rms": float(
            (observations["policy"][:, :21] - clean_observations["policy"][:, :21]).square().mean().sqrt()
        ),
        "teacher_student_joint_max_abs_difference": float(
            (observations["policy"][:, :21] - student_proprio[:, :21]).abs().max()
        ),
        "teacher_privileged_max_abs_error": float(
            (observations["privileged"] - expected).abs().max()
        ),
    }


class _RunnerConfigLoader(yaml.SafeLoader):
    """Read official runner tuples without enabling arbitrary Python objects."""


_RunnerConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def teacher_metadata(checkpoint, object_name, expected_physics=None):
    path = Path(checkpoint).expanduser().resolve(strict=True)
    agent_file = path.parent / "params" / "agent.yaml"
    env_file = path.parent / "params" / "env.yaml"
    agent = yaml.load(agent_file.read_text(), Loader=_RunnerConfigLoader)
    # Env YAML contains Python tuple tags. Read only text here, never execute YAML tags.
    env_text = env_file.read_text()
    if f"/usd/{object_name}/" not in env_text or "revo3_right" not in env_text:
        raise ValueError(f"Teacher asset metadata does not match Revo3 / {object_name}")
    if agent.get("obs_groups", {}).get("actor") != ["policy", "privileged"]:
        raise ValueError("Teacher must use the released Revo3 policy + privileged observation layout")
    if agent.get("actor", {}).get("class_name") != "MLPModel":
        raise ValueError("Only the released RSL-RL 5 MLP teacher is supported")
    protocol_file = path.parent / "params" / "teacher_protocol.json"
    protocol = json.loads(protocol_file.read_text()) if protocol_file.exists() else None
    if protocol is not None:
        from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES

        if (
            protocol.get("quaternion_order") != "xyzw"
            or protocol.get("observation_dims") != {"policy": 49, "privileged": 38}
            or protocol.get("joint_names") != list(REVO3_JOINT_NAMES)
            or protocol.get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
            or protocol.get("environment_sha256") != hashlib.sha256(env_file.read_bytes()).hexdigest()
        ):
            raise ValueError("Teacher XYZW protocol or checkpoint/environment fingerprint is incompatible")
    elif not any(marker in agent.get("experiment_name", "") for marker in ("sim6", "viserdex_revo3_v3")):
        raise ValueError("Teacher must use a Revo3 XYZW checkpoint namespace")
    # BaseLoader treats every YAML value as data, including Python-specific tags.
    physics = yaml.load(env_text, Loader=yaml.BaseLoader).get("sim", {}).get("physics", {}).get("class_type")
    if protocol is not None and protocol.get("physics") != physics:
        raise ValueError("Teacher protocol physics differs from its saved environment")
    if expected_physics is not None and physics != expected_physics:
        raise ValueError(f"Teacher physics {physics!r} differs from rollout physics {expected_physics!r}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "agent": agent,
        "physics": physics,
        "environment_path": str(env_file),
        "environment_sha256": hashlib.sha256(env_file.read_bytes()).hexdigest(),
        "explicit_protocol": protocol,
    }


def load_teacher(metadata, observations, device):
    from rsl_rl.models import MLPModel

    cfg = copy.deepcopy(metadata["agent"]["actor"])
    cfg.pop("class_name")
    model = MLPModel(observations, metadata["agent"]["obs_groups"], "actor", 21, **cfg).to(device)
    payload = torch.load(metadata["path"], map_location=device, weights_only=True)
    if "actor_state_dict" not in payload:
        raise ValueError("Expected an RSL-RL 5 actor_state_dict checkpoint")
    model.load_state_dict(payload["actor_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model
