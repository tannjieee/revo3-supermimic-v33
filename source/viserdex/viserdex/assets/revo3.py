"""BrainCo Revo3 right hand with simulation position drives for in-hand reorientation."""

import math

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .revo3_spec import REVO3_DEFAULT_JOINT_POS, REVO3_JOINT_NAMES, REVO3_MULTIPHYSICS_USD

REVO3_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(REVO3_MULTIPHYSICS_USD),
        activate_contact_sensors=True,
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(),
            PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_depenetration_velocity=1.0,
            ),
        ],
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Local +X is the grasping side, local +Z points along the fingers.
        # Rotate +X upward, placing the palm under the existing object/camera workspace.
        pos=(0.12, 0.0, 0.51),
        rot=(0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)),
        joint_pos=REVO3_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=list(REVO3_JOINT_NAMES),
            # Upstream exports zero drive gains and zero URDF effort/velocity placeholders.
            # These are simulation starting values, not measured hardware specifications.
            joint_effort_limit=0.6,
            joint_velocity_limit=10.0,
            stiffness=7.0,
            damping=0.22,
            friction=0.15,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
