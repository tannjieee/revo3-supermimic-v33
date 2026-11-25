"""Read-only checks against the actual Newton collider and randomized inertia."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def audit_live_hand_convex(asset, env, usd_path):
    """Compare all 144 live convex vertex clouds, in their owning link frames."""
    import newton
    from pxr import Usd, UsdGeom, UsdPhysics
    from .simulation import tensor, collision_shape_mask
    from .geometry_asset import active_colliders, vertices_in
    model = env.sim.physics_manager.get_model()
    stage = Usd.Stage.Open(str(usd_path))
    cache = UsdGeom.XformCache()
    expected = {}
    for prim in active_colliders(stage):
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError(f'Collider has no rigid owner: {prim.GetPath()}')
        cloud = vertices_in(prim, body, cache)
        expected.setdefault(body.GetName(), []).append(cloud)
    mask = collision_shape_mask(asset, model, env.num_envs)
    sources = tensor(asset._root_view.get_attribute('shape_source_ptr', model)).reshape(env.num_envs, -1)
    pointers = set(sources.cpu().numpy()[0, mask[0].cpu().numpy()].tolist())
    flags = tensor(model.shape_flags).cpu().numpy()
    worlds = tensor(model.shape_world).cpu().numpy()
    ids = [i for i, pointer in enumerate(tensor(model.shape_source_ptr).cpu().numpy())
           if int(pointer) in pointers and int(flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)]
    first_world = min(int(worlds[i]) for i in ids)
    ids = [i for i in ids if int(worlds[i]) == first_world]
    if len(ids) != 144:
        raise ValueError(f'Expected 144 live hand hulls in one world, got {len(ids)}')
    transforms = tensor(model.shape_transform).cpu().numpy()
    scales = tensor(model.shape_scale).cpu().numpy()
    kinds = tensor(model.shape_type).cpu().numpy()
    bodies = tensor(model.shape_body).cpu().numpy()
    max_error, vertex_count = 0., 0
    for i in ids:
        if int(kinds[i]) != int(newton.GeoType.CONVEX_MESH):
            raise ValueError('A live hand collision shape is not a convex mesh')
        name = model.body_label[int(bodies[i])].split('/')[-1]
        candidates = expected.get(name, [])
        if not candidates:
            raise ValueError(f'Unexpected live collision owner: {name}')
        points = np.asarray(model.shape_source[i].vertices) * scales[i]
        points = Rotation.from_quat(transforms[i, 3:]).apply(points) + transforms[i, :3]
        tree = cKDTree(points)
        errors = [max(cKDTree(cloud).query(points)[0].max(), tree.query(cloud)[0].max()) for cloud in candidates]
        best = int(np.argmin(errors))
        error = float(errors[best])
        if error > 2e-6:
            raise ValueError(f'Imported manual hull changed on {name}: {error} m')
        candidates.pop(best)
        max_error = max(max_error, error)
        vertex_count += len(points)
    if any(expected.values()):
        raise ValueError('Some authored hand hulls are absent from the live model')
    return dict(collision_instances=144, live_vertex_instances=vertex_count,
                max_vertex_error_m=max_error, comparison_frame='owning rigid link')

def audit_live_convex(asset,env,spec):
    import newton
    from .simulation import tensor, collision_shape_mask
    model=env.sim.physics_manager.get_model()
    mask=collision_shape_mask(asset,model,env.num_envs)
    kinds=tensor(asset._root_view.get_attribute('shape_type',model)).reshape(env.num_envs,-1)[mask]
    scale=tensor(asset._root_view.get_attribute('shape_scale',model)).reshape(env.num_envs,-1,3)[mask]
    assert int(mask[0].sum())==1 and (kinds==int(newton.GeoType.CONVEX_MESH)).all(), kinds
    torch.testing.assert_close(scale,torch.ones_like(scale),atol=1e-7,rtol=0)
    # Match the object view to model mesh pointers, excluding visual-only shapes.
    flags=tensor(model.shape_flags).cpu().numpy()
    sources=tensor(asset._root_view.get_attribute('shape_source_ptr',model)).reshape(env.num_envs,-1).cpu().numpy()[mask.cpu().numpy()]
    pointers=set(sources.tolist())
    ids=[i for i,pointer in enumerate(tensor(model.shape_source_ptr).cpu().numpy())
         if int(pointer) in pointers and int(flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)]
    assert len(ids)==env.num_envs,(len(ids),env.num_envs)
    expected=np.asarray(spec['convex_vertices'])
    transform=tensor(model.shape_transform).cpu().numpy()
    seen=set(); max_error=0.; vertices=[]
    for i in ids:
        mesh=model.shape_source[i]
        if id(mesh) in seen: continue
        seen.add(id(mesh))
        assert mesh is not None
        points=np.asarray(mesh.vertices)
        t=transform[i]
        points=Rotation.from_quat(t[3:]).apply(points)+t[:3]
        error=max(cKDTree(expected).query(points)[0].max(),cKDTree(points).query(expected)[0].max())
        assert error<2e-6,('Imported convex mesh differs from authored geometry',error)
        max_error=max(max_error,float(error));vertices.append(len(points))
    mass=tensor(asset.data.body_mass).reshape(env.num_envs,1)
    inertia=tensor(asset.data.body_inertia).reshape(env.num_envs,3,3)
    reference=torch.tensor(spec['object_inertia'],device=env.device,dtype=inertia.dtype)
    expected_inertia=reference[None]*(mass/spec['object_mass'])[:,:,None]
    assert (mass>=spec['object_mass']*.8-1e-7).all() and (mass<=spec['object_mass']*1.2+1e-7).all()
    torch.testing.assert_close(inertia,expected_inertia,atol=2e-9,rtol=2e-4)
    return dict(shape='convex_mesh',live_mesh_vertex_counts=vertices,collider_count=len(ids),
                errors={'convex_vertices_m':max_error,'convex_inertia_kg_m2':float((inertia-expected_inertia).abs().max())})
