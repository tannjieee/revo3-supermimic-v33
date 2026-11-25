"""Bind the official ray camera to the converted hand's live visual meshes."""

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors import MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg


def create_revo3_hand_depth_camera(cfg: MultiMeshRayCasterCameraCfg):
    """Resolve layered asset geometry after spawning, before physics finalization.

    Official URDF conversion nests links in the kinematic tree. Discover visual
    meshes from that tree instead of assuming the old flat ``link/visuals``
    layout. Each mesh's non-physics frame follows its actual articulated parent.
    """

    robot = sim_utils.find_first_matching_prim(f"{cfg.prim_path}/Robot")
    if robot is None:
        raise RuntimeError("Spawn Revo3 before its hand-depth camera")
    env_path = str(robot.GetPath()).rsplit("/Robot", 1)[0]
    targets = []
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputePurpose() == "guide" or imageable.ComputeVisibility() == "invisible":
            continue
        targets.append(
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=cfg.prim_path + str(prim.GetPath())[len(env_path) :],
                is_shared=True,
                merge_prim_meshes=False,
                track_mesh_transforms=True,
            )
        )
    if not targets:
        raise RuntimeError("Converted Revo3 has no visual meshes for occlusion")
    cfg = cfg.copy()
    cfg.mesh_prim_paths = targets
    cfg.class_type = "isaaclab.sensors:MultiMeshRayCasterCamera"
    return MultiMeshRayCasterCamera(cfg)
