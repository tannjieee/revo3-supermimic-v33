"""Read USD joint frames / real collision geometry without editing assets."""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES


def quaternion(q):
    out = np.array([*q.GetImaginary(), q.GetReal()], dtype=float)
    return (out / np.linalg.norm(out)).tolist()


def mesh_points(prim):
    if prim.IsA(UsdGeom.Mesh):
        return np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
        return np.array(
            [
                [x, y, z]
                for x in (-half, half)
                for y in (-half, half)
                for z in (-half, half)
            ]
        )
    raise ValueError(
        f"Unsupported collision geometry: {prim.GetPath()} ({prim.GetTypeName()})"
    )


def vertices_in(prim, body, cache):
    transform, _ = cache.ComputeRelativeTransform(prim, body)
    return np.array([transform.Transform(Gf.Vec3d(*p)) for p in mesh_points(prim)])


def active_colliders(stage):
    return [
        p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
    ]


def build_geometry_spec(robot_path, object_path, robot_scale=None, object_scale=None):
    for scale in (robot_scale, object_scale):
        if scale is not None and not np.allclose(scale, (1, 1, 1)):
            raise ValueError(
                "Geometry extraction requires scale authored into the validated USD"
            )
    stage = Usd.Stage.Open(str(robot_path))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Expected SI robot asset")
    joints = {
        p.GetName(): UsdPhysics.RevoluteJoint(p)
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.RevoluteJoint)
    }
    if set(joints) != set(REVO3_JOINT_NAMES):
        raise ValueError("Active USD joints differ from policy joint order")
    parents = {str(j.GetBody0Rel().GetTargets()[0]) for j in joints.values()}
    children = {str(j.GetBody1Rel().GetTargets()[0]) for j in joints.values()}
    roots = parents - children
    if len(roots) != 1:
        raise ValueError("Expected a single fixed hand base")
    paths, levels, remaining = [roots.pop()], [], list(REVO3_JOINT_NAMES)
    while remaining:
        group = {key: [] for key in ("parent", "joint", "p0", "r0", "p1", "r1", "axis")}
        ready = [
            name
            for name in remaining
            if str(joints[name].GetBody0Rel().GetTargets()[0]) in paths
        ]
        if not ready:
            raise ValueError("Non-tree USD joint graph")
        next_paths = []
        for name in ready:
            j = joints[name]
            group["parent"].append(paths.index(str(j.GetBody0Rel().GetTargets()[0])))
            group["joint"].append(list(REVO3_JOINT_NAMES).index(name))
            for key, attr in (
                ("p0", j.GetLocalPos0Attr()),
                ("p1", j.GetLocalPos1Attr()),
            ):
                group[key].append(list(attr.Get()))
            group["r0"].append(quaternion(j.GetLocalRot0Attr().Get()))
            group["r1"].append(quaternion(j.GetLocalRot1Attr().Get()))
            group["axis"].append(np.eye(3)["XYZ".index(j.GetAxisAttr().Get())].tolist())
            next_paths.append(str(j.GetBody1Rel().GetTargets()[0]))
            remaining.remove(name)
        paths.extend(next_paths)
        levels.append(group)
    cache = UsdGeom.XformCache()
    root_scale = np.array(
        Gf.Transform(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(paths[0]))
        ).GetScale()
    )
    if not np.allclose(root_scale, 1.0, atol=1e-7):
        raise ValueError(
            "Scaled hand root requires scaled joint frames; current hand must be unit scale"
        )
    clouds = [[] for _ in paths]
    for prim in active_colliders(stage):
        body = prim
        while body and str(body.GetPath()) not in paths:
            body = body.GetParent()
        if body:
            i = paths.index(str(body.GetPath()))
            clouds[i].append(vertices_in(prim, body, cache))
    body_names = [stage.GetPrimAtPath(p).GetName() for p in paths]
    point_body, point_local, point_names = [], [], []

    def add(index, position, suffix):
        point_body.append(index)
        point_local.append(np.asarray(position).tolist())
        point_names.append(body_names[index] + "/" + suffix)

    for i in range(1, len(paths)):
        points = np.concatenate(clouds[i]) if clouds[i] else np.zeros((1, 3))
        add(
            i,
            (points.min(0) + points.max(0)) / 2,
            "collision_center" if clouds[i] else "joint_origin",
        )
        if body_names[i].endswith("_DIP_link"):
            # Mean of distal end vertices, selected along the link's longest dimension.
            axis = np.ptp(points, axis=0).argmax()
            extreme = points[np.argmax(np.abs(points[:, axis])), axis]
            selected = np.abs(points[:, axis] - extreme) <= max(
                np.ptp(points[:, axis]) * 0.1, 1e-7
            )
            add(i, points[selected].mean(0), "distal_surface")
    palm = np.concatenate(clouds[0])
    # Four distributed actual palm surface vertices on the grasp-facing +X side.
    palm = np.unique(palm[palm[:, 0] >= np.quantile(palm[:, 0], 0.65)], axis=0)
    indices = [int(np.argmin(np.linalg.norm(palm - palm.mean(0), axis=1)))]
    for _ in range(3):
        distances = np.linalg.norm(palm[:, None] - palm[indices][None], axis=-1).min(1)
        indices.append(int(distances.argmax()))
    for i, vertex in enumerate(palm[indices]):
        add(0, vertex, f"palm_surface_{i}")
    obj = Usd.Stage.Open(str(object_path))
    colliders = active_colliders(obj)
    if len(colliders) != 1 or not any(colliders[0].IsA(t) for t in (UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Mesh)):
        raise ValueError(
            "Expected exactly one analytic or convex-mesh collider"
        )
    rigid = [p for p in obj.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid) != 1:
        raise ValueError("Expected one rigid object")
    object_cache = UsdGeom.XformCache()
    body_scale = np.array(
        Gf.Transform(object_cache.GetLocalToWorldTransform(rigid[0])).GetScale()
    )
    if not np.allclose(body_scale, body_scale[0], atol=1e-7) or body_scale[0] <= 0:
        raise ValueError(
            "Current object extraction supports positive uniform rigid-root scaling"
        )
    common = dict(version=2, body_names=body_names, levels=levels,
                  point_body=point_body, point_local=point_local, point_names=point_names,
                  object_root_scale=body_scale.tolist())
    if colliders[0].IsA(UsdGeom.Mesh):
        from scipy.spatial import ConvexHull
        if UsdGeom.GetStageMetersPerUnit(obj) != 1 or UsdGeom.GetStageUpAxis(obj) != "Z":
            raise ValueError("Expected SI Z-up convex asset")
        if UsdPhysics.MeshCollisionAPI(colliders[0]).GetApproximationAttr().Get() != "convexHull":
            raise ValueError("Mesh feature extraction requires an authored convex hull")
        points = vertices_in(colliders[0], rigid[0], object_cache) * body_scale[0]
        hull = ConvexHull(points)
        # One normalized supporting plane per face. Merge coplanar triangles
        # only at numerical precision; keep the real rounded collision envelope.
        planes = np.unique(np.round(hull.equations, 8), axis=0)
        planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        mass_api = UsdPhysics.MassAPI(rigid[0])
        mass = float(mass_api.GetMassAttr().Get())
        diag = np.array(mass_api.GetDiagonalInertiaAttr().Get(), dtype=float)
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat(quaternion(mass_api.GetPrincipalAxesAttr().Get())).as_matrix()
        inertia = rotation @ np.diag(diag) @ rotation.T
        if mass <= 0 or (diag <= 0).any() or not np.allclose(body_scale, 1):
            raise ValueError("Convex object requires positive mass/inertia and unit root scale")
        return dict(common, version=3, object_shape="convex_mesh", convex_vertices=points.tolist(),
                    convex_planes=planes.tolist(), convex_triangles=hull.simplices.tolist(),
                    object_mass=mass, object_inertia=inertia.tolist(),
                    object_com=list(mass_api.GetCenterOfMassAttr().Get()),
                    surface_feature="max_normalized_supporting_plane_signed_margin")
    if colliders[0].IsA(UsdGeom.Sphere):
        if UsdGeom.GetStageMetersPerUnit(obj) != 1 or UsdGeom.GetStageUpAxis(obj) != "Z":
            raise ValueError("Expected SI Z-up sphere asset")
        transform, _ = object_cache.ComputeRelativeTransform(colliders[0], rigid[0])
        local_scale = np.array(Gf.Transform(transform).GetScale())
        if not np.allclose(local_scale, local_scale[0], atol=1e-7) or local_scale[0] <= 0:
            raise ValueError("Ellipsoid transforms are not a sphere")
        radius = float(UsdGeom.Sphere(colliders[0]).GetRadiusAttr().Get()) * local_scale[0] * body_scale[0]
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")
        center = np.array(transform.Transform(Gf.Vec3d(0))) * body_scale[0]
        return dict(common, object_shape="sphere", sphere_center=center.tolist(), sphere_radius=radius)
    box = vertices_in(colliders[0], rigid[0], object_cache) * body_scale[0]
    low, high = box.min(0), box.max(0)
    if not np.all(np.isclose(box, low, atol=1e-7) | np.isclose(box, high, atol=1e-7)):
        raise ValueError("Object collision box is not aligned to its body frame")
    return dict(common, object_shape="box", box_center=((low+high)/2).tolist(), box_half=((high-low)/2).tolist())
