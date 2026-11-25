"""Offline contact candidates from the unchanged authored collision hull union.

Sampling approximation only. No physics engine or asset mutation. Candidate
selection is fixed before fitting, without transition labels or target goals.
"""
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, UsdPhysics
from .geometry_asset import active_colliders, vertices_in


def surface_candidates(robot_path, spec, per_link=24, palm_points=128):
    stage=Usd.Stage.Open(str(robot_path));cache=UsdGeom.XformCache()
    clouds={name:[] for name in spec['body_names']}
    com={}
    for prim in stage.Traverse():
        if prim.GetName() in clouds and prim.HasAPI(UsdPhysics.MassAPI):
            com[prim.GetName()]=list(UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get())
    for prim in active_colliders(stage):
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if body.GetName() not in clouds:raise ValueError('Unexpected collision owner')
        clouds[body.GetName()].append(vertices_in(prim,body,cache))
    points=[];owners=[];counts={}
    for i,name in enumerate(spec['body_names']):
        hulls=[ConvexHull(v) for v in clouds[name]]
        candidates=[]
        for h in hulls:
            for ids in h.simplices:
                tri=h.points[ids]
                n=max(1,min(8,int(np.ceil(np.linalg.norm(tri[:,None]-tri[None,:],axis=-1).max()/.004))))
                candidates.extend([(a*tri[0]+b*tri[1]+(n-a-b)*tri[2])/n for a in range(n+1) for b in range(n+1-a)])
        xyz=np.unique(np.round(np.asarray(candidates),8),axis=0)
        # A face covered by another hull of the same rigid body is not exterior.
        interior=np.zeros(len(xyz),dtype=bool)
        for h in hulls:interior|=(xyz@h.equations[:,:3].T+h.equations[:,3]).max(-1)<-2e-7
        xyz=xyz[~interior]
        budget=palm_points if i==0 else per_link
        chosen=[int(np.argmin(np.linalg.norm(xyz-xyz.mean(0),axis=-1)))]
        distance=np.linalg.norm(xyz-xyz[chosen[0]],axis=-1)
        for _ in range(min(budget,len(xyz))-1):
            index=int(distance.argmax());chosen.append(index)
            distance=np.minimum(distance,np.linalg.norm(xyz-xyz[index],axis=-1))
        points.extend(xyz[chosen].tolist());owners.extend([i]*len(chosen));counts[name]=dict(exterior_candidates=len(xyz),selected=len(chosen))
    missing=set(spec['body_names'][1:])-set(com)
    if missing:raise ValueError(f'Missing authored COM: {missing}')
    return dict(point_body=owners,point_local=points,body_com=[com.get(n,[0,0,0]) for n in spec['body_names']],counts=counts,
                protocol='fixed_union_surface_barycentric_farthest_v1',spacing_m=.004,per_link=per_link,palm_points=palm_points)
