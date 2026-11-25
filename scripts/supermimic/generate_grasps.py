"""Generate current-USD grasp candidates; physical certification is a separate command."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from viserdex.assets.prepare_manual_convex import MANUAL_USD
from viserdex.assets.objects import _OBJECT_CFGS
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.supermimic.geometry_asset import build_geometry_spec
from viserdex.supermimic.grasp_candidates import ConvexQueries,propose
from viserdex.supermimic.grasp_bank import asset_contract,digest,validate_record
from viserdex.supermimic.training_limits import load_training_limits
from viserdex.supermimic.identified_hand import load_identified_hand
from viserdex.supermimic.wuji_pose import HOME_POSITION
from viserdex.assets.prepare_manual_convex import inspect_manual_asset,default_manual_source

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--seed_bank',type=Path)
p.add_argument('--candidates',type=int,default=64);p.add_argument('--seed',type=int,default=42)
p.add_argument('--max_nfev',type=int,default=60)
a=p.parse_args()
if a.candidates<1 or a.max_nfev<1: p.error('counts must be positive')
robot=MANUAL_USD;obj=_OBJECT_CFGS['RoundedDie']['usd_path']
spec=build_geometry_spec(robot,obj);limits,_=load_training_limits()
contract=asset_contract(robot,obj,spec,limits)
geometry_limits=np.asarray(contract['geometry_sampling_limits_rad'])
query_spec=dict(spec,joint_names=list(REVO3_JOINT_NAMES));query=ConvexQueries(robot,query_spec)
params,_=load_identified_hand();bias=np.asarray(params['encoder_bias_rad'])
seeds=[dict(q_geometry=list(HOME_POSITION),object_pose_xyzw=[.055,0,.12,0,0,0,1],source_group='fresh-home')]
source=None
if a.seed_bank:
    raw=a.seed_bank.read_bytes();old=json.loads(raw);source=hashlib.sha256(raw).hexdigest()
    for i,r in enumerate(old['records']):
        if 'q_actual' in r:
            # OLD env.reset writes q_actual directly to MuJoCo qpos, with no
            # reference-angle shift in assets._hand_tree: it is geometric q.
            # Its old hold target is deliberately not imported.
            pose=np.asarray(r['object_pose']);pose=np.r_[pose[:3],pose[[4,5,6,3]]]
            seeds.append(dict(q_geometry=r['q_actual'],object_pose_xyzw=pose.tolist(),source_group='legacy-'+str(r.get('source_group_id',r.get('lineage_root_id',i)))))
        else: seeds.append({k:r[k] for k in ('q_geometry','object_pose_xyzw','source_group')})
rng=np.random.default_rng(a.seed);records=[]
a.output.parent.mkdir(parents=True,exist_ok=True)
for i in range(a.candidates):
    seed=seeds[i%len(seeds)]
    r=propose(query,limits,bias,seed['q_geometry'],seed['object_pose_xyzw'],i,rng,a.max_nfev,geometry_limits=geometry_limits)
    r.update(id=f'candidate-{a.seed}-{i}',source_group=seed['source_group'])
    validate_record(r,limits,certified=False,contract={'assets':contract})
    records.append(r)
    out=dict(schema='revo3_grasp_candidates_v1',assets=contract,seed=a.seed,seed_bank_sha256=source,
             method='migrated contact optimization/preload; bounded wrench QP screening; not upstream MALA',records=records)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(candidate=i,geometry_passed=r['geometry_passed'],min_gap=r['query_min_gap_m'],qp=r['qp'])),flush=True)
