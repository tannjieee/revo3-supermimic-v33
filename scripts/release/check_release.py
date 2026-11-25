from pathlib import Path
import copy,hashlib,json,sys,types,torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'source/viserdex'))
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.supermimic.core import Config
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.geometry_asset import build_geometry_spec
from viserdex.supermimic.geometry import geometry_digest
p=torch.load(root/'checkpoints/policy/yellow_actor_32100.pt',map_location='cpu',weights_only=False)
identity=json.loads((root/'configs/rgbd_capture/yellow_policy_identity.json').read_text())
assert hashlib.sha256((root/'checkpoints/policy/yellow_actor_32100.pt').read_bytes()).hexdigest()==identity['checkpoint_sha256']
policy=SavedEnvironmentBinding(root/'configs/rgbd_capture/policy_original_env.yaml')
assert p['metadata']['environment_contract']==policy.contract_sha==identity['environment_contract']
for config in ['configs/rgbd_capture/env.yaml','configs/rgbd_capture/policy_original_env.yaml','configs/yellow_die_handpose/env.yaml']:
    b=SavedEnvironmentBinding(root/config)
    assert all(Path(a['path']).is_relative_to(root) for a in b.assets.values())
    value=copy.deepcopy(b.saved)
    for name in ['robot','object']:
        value['scene'][name]['spawn']['usd_path']=b.assets[name]['path']
    goal=value['commands']['object_pose']['goal_pose_visualizer_cfg']['markers']['goal']
    goal['usd_path']=b.asset_paths.get(goal['usd_path'],goal['usd_path'])
    scene=types.SimpleNamespace(**{
        n:types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=b.assets[n]['path']))
        for n in ['robot','object']})
    solver=types.SimpleNamespace(**{n:value['sim']['physics']['solver_cfg'][n]
                                  for n in ['nconmax','njmax']})
    physics=types.SimpleNamespace(class_type=value['sim']['physics']['class_type'], solver_cfg=solver)
    cfg=types.SimpleNamespace(to_dict=lambda:value, scene=scene,
        sim=types.SimpleNamespace(physics=physics, dt=value['sim']['dt']), decimation=value['decimation'])
    b.verify(cfg)
    value['decimation']+=1
    try: b.verify(cfg)
    except ValueError: pass
    else: raise AssertionError('Dynamics tampering must fail')
    value['decimation']-=1
    value['scene']['robot']['spawn']['usd_path']='/unapproved/path'
    try: b.verify(cfg)
    except ValueError: pass
    else: raise AssertionError('Unapproved path must fail')
actor=Actor(Config(**p['config'])).eval()
actor.load_state_dict(p['actor'],strict=True)
spec=build_geometry_spec(policy.assets['robot']['path'],policy.assets['object']['path'])
assert geometry_digest(spec)==p['metadata']['geometry_digest']
print('PASS: relocated asset geometry digest matches Actor metadata.')
print('PASS: Actor strict load, policy SHA/contract, three relocated configs, changed dynamics/path rejection.')
