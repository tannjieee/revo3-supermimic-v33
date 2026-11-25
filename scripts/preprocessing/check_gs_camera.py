# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth, Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to extract data to generate a gaussian splat model.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
import isaaclab.sim as sim_utils
import omni.log
import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer
from viserdex.renderer import GaussianSplatCameraCfg
from viserdex.collectors.location_sampler_cfg import IcoSphereLocationSamplerCfg

from viserdex.assets.objects import TARGET_OBJECT_CFG as object_cfg


# Configuration for Gaussian Splat Camera
@configclass
class GSCameraCubeSceneCfg(InteractiveSceneCfg):
    # object
    object_0 = RigidObjectCfg(
        prim_path="/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_cfg["usd_path"],
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # rigid object to attach the cameras to
    base: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/base",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=PhysxRigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=PhysxRigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.0, 0.0)),
    )

    camera_0 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/rgb_cam",
        update_period=0,
        data_types=["rgb", "distance_to_image_plane"],
        height=480,
        width=848,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1e6),
        ),
        debug_vis=True,
    )

    gsplat_camera = GaussianSplatCameraCfg(
        prim_path="{ENV_REGEX_NS}/base",
        update_period=0,
        data_types=["rgb", "distance_to_image_plane"],
        camera_model=GaussianSplatCameraCfg.PinholeCfg(
            horizontal_aperture=20.955,
            focal_length=24.0,
            height=480,
            width=848,
        ),
        splat_model_path=object_cfg["splat_path"],
        clipping_range=(0.1, 1e10),
        augmentations=[],
        debug_vis=True,
    )

    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device or "cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((-1.3, 0.0, 0.4), (1.1, 0.0, -0.4))

    # Initialize scene
    scene_cfg = GSCameraCubeSceneCfg(args_cli.num_envs, env_spacing=1.0)
    with Timer("[INFO]: Scene creation time", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    omni.log.info(f"Scene manager: {scene}")
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    # Sample explicit XYZW world poses. Both renderers use the same camera pose;
    # the splat renderer then expresses it in the object's local frame.
    sampling_cfg = IcoSphereLocationSamplerCfg(radius=0.25, subdivisions=0)
    samples = sampling_cfg.func(sampling_cfg)
    indices = np.random.choice(len(samples), args_cli.num_envs, replace=False)
    camera_poses = samples[indices].to(sim.device)
    object_pos = torch.zeros(1, 3, device=sim.device)
    object_quat = math_utils.random_orientation(1, device=sim.device)
    scene.rigid_objects["object_0"].write_root_pose_to_sim_index(root_pose=torch.cat([object_pos, object_quat], dim=-1))
    # Publish the rigid-body pose to Fabric before the RTX capture. A render-only
    # update does not publish teleported rigid transforms in this Sim 6 backend.
    sim.step(render=False)
    sim.forward()
    scene.update(sim.get_physics_dt())
    scene.sensors["camera_0"].set_world_poses(camera_poses[:, :3], camera_poses[:, 3:], convention="world")
    camera_pos_frame_object, camera_quat_frame_object = math_utils.subtract_frame_transforms(
        object_pos.expand(args_cli.num_envs, -1),
        object_quat.expand(args_cli.num_envs, -1),
        camera_poses[:, :3],
        camera_poses[:, 3:],
    )
    scene.sensors["gsplat_camera"].set_world_poses(
        camera_pos_frame_object, camera_quat_frame_object, convention="world"
    )

    # execute some render steps to make sure the buffers are filled
    for _ in range(10):
        sim.render()

    # update the scene
    scene.update(sim.get_physics_dt())

    # get the rendered images of normal rgb and gsplat rgb
    rgb_cam_output = scene.sensors["camera_0"].data.output["rgb"].torch  # [B, H, W, 3]
    gsplat_cam_output = scene.sensors["gsplat_camera"].data.output["rgb"].torch  # [B, H, W, 3]

    for name, rgb in (("RTX", rgb_cam_output), ("GS", gsplat_cam_output)):
        assert (rgb[..., :3].amax(dim=-1) > 20).sum() > 100, f"{name} rendered no visible object"
    # create a matplotlib figure to display the images next to each other
    fig, axs = plt.subplots(args_cli.num_envs, 2, squeeze=False)
    for i in range(args_cli.num_envs):
        axs[i, 0].imshow(rgb_cam_output[i].cpu().numpy())
        axs[i, 1].imshow(gsplat_cam_output[i].cpu().numpy())

        if i == 0:
            axs[i, 0].set_title("RGB Camera")
            axs[i, 1].set_title("Gaussian Splat RGB Camera")
    # save the figure in an output directory at the same level as the script
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "gsplat_rgb_camera_comparison.png"))
    plt.close("all")

    # get the rendered images of distance to image plane
    depth_cam_output = scene.sensors["camera_0"].data.output["distance_to_image_plane"].torch  # [B, H, W, 1]
    depth_gsplat_cam_output = (
        scene.sensors["gsplat_camera"].data.output["distance_to_image_plane"].torch.clone()
    )  # [B, H, W, 1]
    depth_gsplat_cam_output[depth_gsplat_cam_output == 0] = float("inf")

    finite_depth = torch.cat(
        [
            depth_cam_output[torch.isfinite(depth_cam_output)],
            depth_gsplat_cam_output[torch.isfinite(depth_gsplat_cam_output)],
        ]
    )
    if not finite_depth.numel():
        raise RuntimeError("Neither camera rendered valid foreground depth.")
    vmin, vmax = finite_depth.min().item(), finite_depth.max().item()

    # create a matplotlib figure to display the images next to each other
    fig, axs = plt.subplots(args_cli.num_envs, 2, squeeze=False)
    for i in range(args_cli.num_envs):
        axs[i, 0].imshow(depth_cam_output[i].cpu().numpy(), vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(depth_gsplat_cam_output[i].cpu().numpy(), vmin=vmin, vmax=vmax)

        if i == 0:
            axs[i, 0].set_title("Depth (RTX Cam)")
            axs[i, 1].set_title("Depth (Gaussian Splat Cam)")
    # save the figure in output directory
    print(f"[INFO] Saving output images to {output_dir}")
    fig.savefig(os.path.join(output_dir, "gsplat_distance_to_image_plane_comparison.png"))
    plt.close("all")

    print("[INFO] Done: both comparison images written.")


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.app.post_quit(exit_code)
        simulation_app.close()
