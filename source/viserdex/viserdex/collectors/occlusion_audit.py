"""Independent visual-mesh ray audit, with no added bodies or simulation edits."""
import numpy as np
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _object_depth(mesh: wp.uint64, k: wp.array(dtype=wp.mat33),
                  rotations: wp.array(dtype=wp.mat33), translations: wp.array(dtype=wp.vec3),
                  depth: wp.array3d(dtype=float)):
    e, y, x = wp.tid()
    # Match the pinned Newton OpenCV camera's pixel-centre rays exactly.
    ray = wp.vec3((float(x) + 0.5 - k[e][0, 2]) / k[e][0, 0],
                  (float(y) + 0.5 - k[e][1, 2]) / k[e][1, 1], 1.0)
    inv_r = wp.transpose(rotations[e])
    origin = -(inv_r * translations[e])
    direction = inv_r * wp.normalize(ray)
    hit = wp.mesh_query_ray(mesh, origin, direction, 3.0 * wp.length(ray))
    depth[e, y, x] = 0.0
    if hit.result:
        depth[e, y, x] = hit.t / wp.length(ray)


class OcclusionAudit:
    def __init__(self, usd_path, scale, device):
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(str(usd_path))
        cache = UsdGeom.XformCache()
        vertices, indices, paths = [], [], []
        count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputePurpose() == 'guide' or imageable.ComputeVisibility() == 'invisible':
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            transform = np.asarray(cache.GetLocalToWorldTransform(prim))
            points = points @ transform[:3, :3] + transform[3, :3]
            points *= np.asarray(scale or (1., 1., 1.))
            sizes = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            assert np.all(sizes == 3), 'Audit requires triangulated visual mesh'
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            vertices.append(points.astype(np.float32))
            indices.append(faces + count)
            count += len(points)
            paths.append(str(prim.GetPath()))
        assert vertices, 'No visible mesh found for audit'
        self.mesh = wp.Mesh(points=wp.array(np.concatenate(vertices), dtype=wp.vec3, device=device),
                            indices=wp.array(np.concatenate(indices), dtype=wp.int32, device=device))
        self.contract = dict(method='independent Warp raycast against the same visual USD mesh',
            visual_meshes=paths, vertices=count, triangles=sum(len(i)//3 for i in indices),
            added_simulation_bodies=0, depth_tolerance_m=0.0005,
            minimum_visible_depth_pixels=200, minimum_visible_rgb_pixels=32,
            ray_pixel_centres='x+0.5, y+0.5, matching pinned Newton OpenCV renderer',
            audit_depth_is_input=False)

    def render(self, k, rotation, translation):
        depth = torch.zeros((len(k), 480, 848), dtype=torch.float32, device=k.device)
        wp.launch(_object_depth, dim=depth.shape, inputs=[self.mesh.id,
            wp.from_torch(k.contiguous(), dtype=wp.mat33),
            wp.from_torch(rotation.contiguous(), dtype=wp.mat33),
            wp.from_torch(translation.contiguous(), dtype=wp.vec3), wp.from_torch(depth)],
            device=str(k.device))
        return depth

    def check(self, depth, visible, rgb_mask, isolated):
        silhouette = isolated > 0
        overlap = visible & silhouette
        disagreement = overlap & (torch.abs(depth-isolated) > self.contract['depth_tolerance_m'])
        outside = visible & ~silhouette
        reasons = []
        if int(visible.sum()) < 200:
            reasons.append('insufficient_visible_depth')
        if int(rgb_mask.sum()) < 32:
            reasons.append('insufficient_visible_rgb')
        if int(disagreement.sum()):
            reasons.append('first_hit_depth_conflict')
        if int(outside.sum()):
            reasons.append('target_outside_isolated_silhouette')
        return dict(passed=not reasons, reasons=reasons,
            silhouette_pixels=int(silhouette.sum()), visible_depth_pixels=int(visible.sum()),
            conflicting_pixels=int(disagreement.sum()), outside_silhouette_pixels=int(outside.sum()),
            max_visible_depth_error_m=float(torch.abs(depth[overlap]-isolated[overlap]).max()) if overlap.any() else 0.,
            occlusion_fraction=float((silhouette & ~visible).sum()) / max(1, int(silhouette.sum())))
