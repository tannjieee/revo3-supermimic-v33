"""Revo3 port of the ViserDex in-hand task and its pose-estimator variants."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from viserdex.assets.revo3 import REVO3_HAND_CFG
from viserdex.assets.revo3_spec import REVO3_COMPACT_USD, REVO3_PALM_SDF_USD, REVO3_V14_TRAINING_USD
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.tasks.manipulation.inhand import inhand_env_cfg, mdp
from viserdex.tasks.manipulation.inhand.physics_cfg import (
    InHandHikrobotPhysicsCfg, InHandSDFPhysicsCfg, InHandV14PhysicsCfg,
)


@configclass
class ViserDexRevo3EnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = REVO3_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Validated on a 32 GiB RTX 5090 with the MJWarp contact capacity below.
        self.scene.num_envs = 2048

        # Same explicit ordering for all joint-valued policy inputs and outputs.
        self.actions.joint_pos.joint_names = list(REVO3_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        for term in (self.observations.policy.joint_pos, self.observations.privileged.joint_vel):
            term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(REVO3_JOINT_NAMES), preserve_order=True)

        # Reset near a Revo3-specific open grasp, rather than applying the Allegro
        # fraction-of-range reset to every joint (including thumb opposition and splay).
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(REVO3_JOINT_NAMES), preserve_order=True),
            "position_range": (-0.03, 0.03),
            "velocity_range": (0.0, 0.0),
        }
        self.sim.default_visualizer_cfg.eye = (-0.35, -0.35, 0.85)
        self.sim.default_visualizer_cfg.lookat = (0.02, 0.0, 0.565)


@configclass
class ViserDexRevo3HikrobotEnvCfg(ViserDexRevo3EnvCfg):
    """Native full-scene RGB from the same calibrated camera as the preview video.

    This does not create the legacy Gaussian camera or an image-to-pose term;
    the student entrypoint runs its frozen CNN explicitly on this sensor's RGB.
    """

    def __post_init__(self):
        super().__post_init__()
        from viserdex.renderer.hikrobot_calibration import DEFAULT_CALIBRATION_DIRECTORY, make_calibrated_camera

        self.scene.num_envs = 1
        self.scene.robot.spawn.usd_path = str(REVO3_COMPACT_USD)
        self.sim.physics = InHandHikrobotPhysicsCfg()
        self.sim.dt = 1 / 120
        self.decimation = 4
        self.sim.render_interval = 4
        self.commands.object_pose.debug_vis = False
        initial = self.scene.robot.init_state
        self.scene.hikrobot_camera = make_calibrated_camera(
            DEFAULT_CALIBRATION_DIRECTORY, initial.pos, initial.rot,
        )
        self.sim.default_visualizer_cfg.eye = (0.67, -0.45, 0.98)
        self.sim.default_visualizer_cfg.lookat = (0.105, 0.025, 0.61)


@configclass
class ViserDexRevo3SDFEnvCfg(ViserDexRevo3EnvCfg):
    """Revo3 teacher with one non-convex SDF replacing only the fixed palm."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(REVO3_PALM_SDF_USD)
        self.sim.physics = InHandSDFPhysicsCfg()


@configclass
class ViserDexRevo3V14EnvCfg(ViserDexRevo3EnvCfg):
    """Keep the teacher/control contract while replacing only the hand collision asset."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(REVO3_V14_TRAINING_USD)
        self.sim.physics = InHandV14PhysicsCfg()


@configclass
class ViserDexRevo3PoseEstimatorEnvCfg(ViserDexRevo3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        inhand_env_cfg.pose_estimator_post_init(self)
        # Paper III-C2: track the actual Revo3 link geometry and compare its
        # optical-axis depth against the object-only Gaussian rendering.
        self.observations.pose.image_features.params["masking_data"] = None
        gs = self.scene.gsplat_camera
        camera = gs.camera_model
        self.scene.hand_depth_camera = MultiMeshRayCasterCameraCfg(
            # Ray-cast sensors attach to an existing frame; the environment
            # root stays fixed even while the GS camera moves in object space.
            prim_path="{ENV_REGEX_NS}",
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=gs.offset.pos,
                rot=gs.offset.rot,
                convention=gs.offset.convention,
            ),
            class_type="viserdex.renderer.revo3_hand_depth_camera:create_revo3_hand_depth_camera",
            mesh_prim_paths=[],  # Resolved from the generated layered asset after spawning.
            pattern_cfg=PinholeCameraPatternCfg(
                width=camera.width,
                height=camera.height,
                focal_length=camera.focal_length,
                horizontal_aperture=camera.horizontal_aperture,
                vertical_aperture=camera.vertical_aperture,
                horizontal_aperture_offset=camera.horizontal_aperture_offset,
                vertical_aperture_offset=camera.vertical_aperture_offset,
            ),
            data_types=["distance_to_image_plane"],
            update_period=0.0,
            max_distance=1.0,
            depth_clipping_behavior="none",
        )
        self.observations.pose.image_features.params["hand_camera_cfg"] = SceneEntityCfg("hand_depth_camera")


@configclass
class ViserDexRevo3PoseEstimatorEvalEnvCfg(ViserDexRevo3PoseEstimatorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        inhand_env_cfg.pose_estimator_eval_post_init(self)
        self.scene.robot.spawn.visible = False


@configclass
class ViserDexRevo3PoseEstimatorCameraEnvCfg(ViserDexRevo3PoseEstimatorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        inhand_env_cfg.pose_estimator_camera_post_init(self)
        self.scene.robot.spawn.visible = False


@configclass
class ViserDexRevo3PoseEstimatorUnaugmentedEnvCfg(ViserDexRevo3PoseEstimatorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        inhand_env_cfg.pose_estimator_unaugmented_post_init(self)


@configclass
class ViserDexRevo3EnvCfg_PLAY(ViserDexRevo3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
