"""Restore the saved teacher environment before the official launcher starts."""

import copy
import hashlib
import json
from pathlib import Path

import yaml


class EnvironmentLoader(yaml.SafeLoader):
    """Only the inert tuple/slice values emitted by official config serialization."""


EnvironmentLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
EnvironmentLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def load_environment(path):
    return yaml.load(Path(path).read_text(), Loader=EnvironmentLoader)


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    return value


def _remove_spawn_paths(value):
    # Spawners fill this output field after creating a scene. It is not an
    # input asset path and must be regenerated for the new environment count.
    if isinstance(value, dict):
        value.pop("spawn_path", None)
        for child in value.values():
            _remove_spawn_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _remove_spawn_paths(child)


def environment_contract(data):
    """Dynamics/observation contract; operational settings are recorded separately."""
    value = copy.deepcopy(data)
    _remove_spawn_paths(value)
    # Version-25 YAML predates this optional discriminator. Its default is an
    # exact legacy behavior, so omit only that default from contract hashing.
    action = value.get("actions", {}).get("joint_pos", {})
    if action.get("limit_contract") == "legacy":
        action.pop("limit_contract")
    # New direct-action fields preserve historical behavior only at exact defaults.
    if action.get("dex_reference_profile") is None:
        action.pop("dex_reference_profile", None)
        if action.get("alpha") == 1.0 and 'direct_actions' in str(action.get('class_type', '')):
            action.pop("alpha")
    command = value.get("commands", {}).get("object_pose", {})
    # New reward fields at exact disabled defaults do not change old YAML hashes.
    from viserdex.supermimic.quiet_hold import LEGACY_DEFAULTS
    for name, default in LEGACY_DEFAULTS.items():
        if command.get(name) == default:
            command.pop(name)
    # Historical YAML omitted this field and tracked position. Only that exact
    # default is hash-equivalent; orientation-only holding is a new contract.
    if command.get("track_object_position") is True:
        command.pop("track_object_position")
    for key in ("seed", "log_dir", "ui_window_class_type", "rerender_on_reset", "num_rerenders_on_reset",
                "video_recorders", "recorders", "export_io_descriptors", "xr"):
        value.pop(key, None)
    sim = value["sim"]
    for key in ("device", "log_dir", "logging_level", "save_logs_to_file", "visualizer_cfgs", "default_visualizer_cfg"):
        sim.pop(key, None)
    solver = sim["physics"].get("solver_cfg", {})
    for key in ("nconmax", "njmax"):
        solver.pop(key, None)
    scene = value["scene"]
    for key in ("num_envs", "clone_in_fabric"):
        scene.pop(key, None)
    for asset in ("robot", "object"):
        scene[asset]["spawn"].pop("activate_contact_sensors", None)
    for name in ("policy", "privileged"):
        value["observations"][name]["enable_corruption"] = False
    value["commands"]["object_pose"]["debug_vis"] = False

    def remove_resolved_ids(item):
        if isinstance(item, dict):
            for names, ids in (("joint_names", "joint_ids"), ("body_names", "body_ids"),
                               ("fixed_tendon_names", "fixed_tendon_ids")):
                if item.get(names) is not None:
                    item.pop(ids, None)
            for child in item.values():
                remove_resolved_ids(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                remove_resolved_ids(child)

    remove_resolved_ids(value)
    return _json_value(value)


def contract_sha256(data):
    return hashlib.sha256(json.dumps(environment_contract(data), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SavedEnvironmentBinding:
    def __init__(self, path, expected_sha256=None):
        self.path = Path(path).resolve(strict=True)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Saved environment YAML SHA-256 changed")
        self.saved = load_environment(self.path)
        self.contract = environment_contract(self.saved)
        self.contract_sha = contract_sha256(self.saved)
        self.assets = {}
        # Keep the original contract identity while resolving identical bundled
        # assets in this checkout. Never remap arbitrary external paths.
        self.asset_paths = {}
        checkout = Path(__file__).resolve().parents[4]
        originals = [self.saved["scene"][name]["spawn"]["usd_path"]
                     for name in ("robot", "object")]
        originals.append(self.saved["commands"]["object_pose"][
            "goal_pose_visualizer_cfg"]["markers"]["goal"]["usd_path"])
        for original in originals:
            marker = "/repository/"
            if marker in original:
                relative = original.split(marker, 1)[1]
                if relative.startswith(("assets/", "source/viserdex/viserdex/assets/")):
                    local = (checkout / relative).resolve(strict=True)
                    if not local.is_relative_to(checkout):
                        raise ValueError("Asset relocation escaped the checkout")
                    self.asset_paths[original] = str(local)
        for name in ("robot", "object"):
            original = self.saved["scene"][name]["spawn"]["usd_path"]
            path = Path(self.asset_paths.get(original, original)).resolve(strict=True)
            self.assets[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def configure(self, cfg, args):
        expected = self.saved["sim"]["physics"]["class_type"]
        if str(cfg.sim.physics.class_type) != expected:
            raise ValueError("Selected physics backend differs from the saved teacher environment")
        data = copy.deepcopy(self.saved)
        _remove_spawn_paths(data)
        data.pop("seed", None)  # Runtime seed is independently selected and recorded.
        data.pop("log_dir", None)
        data["sim"].pop("log_dir", None)
        for name in ("robot", "object"):
            spawn = data["scene"][name]["spawn"]
            spawn["usd_path"] = self.asset_paths.get(spawn["usd_path"], spawn["usd_path"])
        goal = data["commands"]["object_pose"]["goal_pose_visualizer_cfg"]["markers"]["goal"]
        goal["usd_path"] = self.asset_paths.get(goal["usd_path"], goal["usd_path"])
        # Public configclass API preserves the task's typed schema/callables.
        if hasattr(cfg.commands.object_pose, 'track_object_position'):
            data['commands']['object_pose'].setdefault('track_object_position', True)
        # configclass recursively updates dictionaries and requires their keys
        # to exist; v33 is an optional schema absent from historical tasks.
        profile = data.get('actions', {}).get('joint_pos', {}).get('dex_reference_profile')
        if profile is not None and hasattr(cfg.actions.joint_pos, 'dex_reference_profile'):
            cfg.actions.joint_pos.dex_reference_profile = copy.deepcopy(profile)
        cfg.from_dict(data)
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device or "cuda:0"
        for override in args.preset_overrides:
            key, value = override.split("=", 1)
            if key in ("env.sim.physics.solver_cfg.nconmax", "env.sim.physics.solver_cfg.njmax"):
                name = key.rsplit(".", 1)[1]
                capacity = int(value)
                if capacity < self.saved["sim"]["physics"]["solver_cfg"][name]:
                    raise ValueError("Do not reduce the saved teacher collision capacity")
                setattr(cfg.sim.physics.solver_cfg, name, capacity)
            elif key != "physics":
                raise ValueError(f"Teacher-bound rollout refuses unsaved override: {key}")
        self.verify(cfg)

    def verify(self, cfg):
        actual = environment_contract(cfg.to_dict())
        # Validate the relocated path before restoring its canonical contract
        # spelling. Other dynamics/observation/geometry checks remain strict.
        for name in ("robot", "object"):
            spawn = actual["scene"][name]["spawn"]
            original = self.contract["scene"][name]["spawn"]["usd_path"]
            if spawn["usd_path"] == self.asset_paths.get(original, original):
                spawn["usd_path"] = original
        goal = actual["commands"]["object_pose"]["goal_pose_visualizer_cfg"]["markers"]["goal"]
        original = self.contract["commands"]["object_pose"]["goal_pose_visualizer_cfg"]["markers"]["goal"]["usd_path"]
        if goal["usd_path"] == self.asset_paths.get(original, original):
            goal["usd_path"] = original
        if actual != self.contract:
            changed = [key for key in set(actual) | set(self.contract) if actual.get(key) != self.contract.get(key)]
            raise ValueError(f"Saved teacher environment contract differs in: {sorted(changed)}")
        for name, record in self.assets.items():
            path = Path(getattr(cfg.scene, name).spawn.usd_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Teacher {name} asset SHA-256 changed")
        return {
            "source_environment": str(self.path), "source_environment_sha256": self.sha256,
            "contract_sha256": self.contract_sha, "assets": self.assets,
            "physics": str(cfg.sim.physics.class_type), "step_dt": cfg.sim.dt * cfg.decimation,
            "nconmax": cfg.sim.physics.solver_cfg.nconmax, "njmax": cfg.sim.physics.solver_cfg.njmax,
        }
