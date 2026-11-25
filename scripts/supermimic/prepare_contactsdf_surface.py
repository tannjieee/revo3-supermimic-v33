"""Fixed surface sampling plus geometry-only closest points at the declared HOME.

Closest points use the same collision assets, not simulator transition labels.
No model fitting, test data, asset edits, settling or runtime contact oracle.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.supermimic.contact_sdf_assets import surface_candidates
from viserdex.supermimic.core import Config
from viserdex.supermimic.support_placement import SupportPlacement
from viserdex.supermimic.fixed_palm_reset import POSITION_BASE_M,QUATERNION_BASE_XYZW


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    payload=torch.load(a.checkpoint,weights_only=False,map_location='cpu');cfg=Config(**payload['config']);spec=payload['geometry_spec']
    robot=payload['metadata']['environment_binding']['assets']['robot']['path'];surface=surface_candidates(robot,spec)
    query=SupportPlacement(robot,spec);query.set_pose(np.array(cfg.action_home),POSITION_BASE_M,QUATERNION_BASE_XYZW)
    anchors=[]
    for g in query.hand_geoms:
        pts=np.zeros(6);gap=query.mujoco.mj_geomDistance(query.model,query.data,g,query.object_id,.01,pts)
        if gap>.002:continue
        body_id=query.model.geom_bodyid[g];name=query.model.body(body_id).name.removeprefix('query_');body=spec['body_names'].index(name)
        local=(pts[:3]-query.data.xpos[body_id])@query.data.xmat[body_id].reshape(3,3)
        surface['point_body'].append(body);surface['point_local'].append(local.tolist());anchors.append(dict(body=name,gap_m=float(gap),point_local=local.tolist()))
    surface['home_geometry_anchors']=anchors;surface['protocol']+='_home_geometry_anchors';surface['checkpoint_geometry_source']=str(a.checkpoint)
    a.output.write_text(json.dumps(surface,indent=2)+'\n');print('surface',len(surface['point_body']),'HOME anchors',len(anchors))
if __name__=='__main__':main()
