"""Physics choices following Isaac Lab 3.0's official PresetCfg contract."""

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class InHandPhysicsCfg(PresetCfg):
    """Keep engine settings separate from the shared task and asset definitions.

    MJWarp capacity is per environment, unlike PhysX's global GPU buffers.
    Size contact and constraint storage from the measured reset/action workload;
    do not translate PhysX capacities numerically. The 2048-world Revo3 audit
    observed <= 444 broadphase pairs/world on average (the contact pool is
    shared) and <= 474 constraint rows in any world. Leave headroom and keep
    overflow diagnostics enabled as the reset/policy distribution expands.
    """

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**22,
        gpu_max_rigid_patch_count=2**23,
        gpu_collision_stack_size=2**26,
        enable_ccd=False,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            njmax=1024,
            nconmax=640,
            # A 2048-world / 1200-step audit reached the upstream 50-step
            # line-search cap. This budget passed the same workload.
            ls_iterations=100,
            cone="elliptic",
            update_data_interval=1,
            use_mujoco_contacts=True,
        ),
        num_substeps=2,
        debug_mode=True,
    )
    default = newton_mjwarp


@configclass
class InHandHikrobotPhysicsCfg(InHandPhysicsCfg):
    """Compact-hand single/few-world RGB playback, matching the measured preview."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast", njmax=512, nconmax=2048,
            ls_iterations=100, cone="elliptic", update_data_interval=1,
            use_mujoco_contacts=True,
        ),
        use_cuda_graph=False,
        num_substeps=2,
        debug_mode=True,
    )
    default = newton_mjwarp


@configclass
class InHandSDFPhysicsCfg(InHandPhysicsCfg):
    """Official Newton contacts are required for the non-convex palm SDF."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Newton's mesh/SDF pipeline observed up to 724 contacts/world
            # on average in the shared pool; 640 truncated that workload.
            integrator="implicitfast",
            njmax=1024,
            nconmax=1024,
            ls_iterations=100,
            cone="elliptic",
            update_data_interval=1,
            use_mujoco_contacts=False,
        ),
        # 4096 worlds produced 16.7M candidate triangle pairs. The default
        # 1M global buffer truncated collisions; retain ~2x measured headroom.
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="explicit",
            reduce_contacts=True,
            max_triangle_pairs=2**25,
        ),
        # Detection-only proximity band, per shape. A 5 mm override was faster
        # but added high-speed thin-wall misses in the isolated palm audit.
        # Keep the conservative 10 mm baseline; neither setting provides CCD.
        default_shape_cfg=NewtonShapeCfg(gap=0.01),
        num_substeps=2,
        debug_mode=True,
    )
    default = newton_mjwarp


@configclass
class InHandV14PhysicsCfg(InHandPhysicsCfg):
    """V14 convex contacts with measured buffer budgets and the teacher's time step."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast", njmax=512, nconmax=768,
            ls_iterations=100, cone="elliptic", update_data_interval=1,
            use_mujoco_contacts=True,
        ),
        default_shape_cfg=NewtonShapeCfg(gap=0.01),
        num_substeps=2,
        debug_mode=True,
    )
    default = newton_mjwarp
