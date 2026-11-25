"""Compare paired physical-angle finite differences with a frozen WM."""
import sys, json, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'source/viserdex'))
from viserdex.supermimic.core import Config, action_target, advance, rotation_log, quat_mul, quat_conjugate, reward
from viserdex.supermimic.networks import Dynamics
from viserdex.supermimic.state_layout import COMPACT

torch.set_num_threads(4)
folder, checkpoint = Path(sys.argv[1]), Path(sys.argv[2])
d = {k: torch.load(folder/(k+'.pt'), weights_only=False) for k in ('base','repeat','plus','minus')}
replicated=(folder/'repeat3.pt').exists()
if replicated:
    d.update({k:torch.load(folder/(k+'.pt'),weights_only=False) for k in ('repeat2','repeat3','plus2','minus2')})
p = torch.load(checkpoint, weights_only=False, map_location='cpu'); cfg=Config(**p['config'])
wm=Dynamics(d['base']['context'].shape[-1],cfg).eval(); wm.load_state_dict(p['wm'])
ctx=d['base']['context']; limits=d['base']['limits']; start=d['base']['probe_step']
report={'checkpoint':str(checkpoint),'worlds':len(ctx),'eps_deg':d['plus']['eps_deg'], 'seed':d['base'].get('seed',44), 'source_step':d['base'].get('source_step',128)}
report['prefix_max_abs']={k:max((r['next_physical']-b['next_physical']).abs().max().item() for r,b in zip(v['records'][:start],d['base']['records'][:start])) if start else (v['records'][0]['state']-d['base']['records'][0]['state']).abs().max().item() for k,v in d.items()}
if max(report['prefix_max_abs'].values()) != 0:
    report['accepted']=False; (folder/'response.json').write_text(json.dumps(report,indent=2))
    raise RuntimeError('Different starting trajectories: directional comparison rejected')
report['repeat_max_abs']=max((r['next_physical']-b['next_physical']).abs().max().item() for r,b in zip(d['repeat']['records'],d['base']['records']))
if report['repeat_max_abs']!=0 and not replicated:
    report['accepted']=False; (folder/'response.json').write_text(json.dumps(report,indent=2))
    raise RuntimeError('Zero perturbation replay differs: directional comparison rejected')
report['accepted']=True
report['interpretation']='Finite perturbation response with repeated-solver noise filtering; not a full infinitesimal Jacobian' if replicated else 'Exact replay'
report['resolved_rule']='response norm > 10 * max(control and same-sign repeat rotation differences, 1e-6 rad)' if replicated else None

def rotation(a,b): return rotation_log(quat_mul(a[:,45:49],quat_conjugate(b[:,45:49])))
def stats(real,model,mask):
    r,m=real[mask],model[mask]; rn=r.norm(dim=-1); mn=m.norm(dim=-1)
    active=rn>1e-5
    cos=(r*m).sum(-1)/(rn*mn).clamp_min(1e-20)
    return dict(samples=len(r),resolved_samples=int(active.sum()),real_response_deg_median=rn.median().item()*180/math.pi,
        model_response_deg_median=mn.median().item()*180/math.pi,
        cosine_mean=cos[active].mean().item(),positive_alignment_fraction=(cos[active]>0).float().mean().item(),
        relative_vector_error=(m-r).norm().item()/rn.norm().clamp_min(1e-20).item())
with torch.no_grad():
    states={k:d[k]['records'][start]['state'].clone() for k in ('plus','minus','base')}
    mask=(d['plus']['records'][start]['action']-d['minus']['records'][start]['action']).abs().amax(-1)>0
    report['eligible_worlds']=int(mask.sum())
    report['horizons']={}
    for t in range(8):
        pred={}
        before={k:v for k,v in states.items()}
        for k in states:
            row=d[k]['records'][start+t]; a=row['action']; target=action_target(states[k],a,limits,cfg)
            pred[k]=wm(states[k],target,ctx)
            states[k]=advance(states[k],pred[k],a,target,states[k][:,COMPACT.goal_rot],500,hold_cfg=cfg)
            mask &= ~row['done']
        for branch in d.values():
            mask &= ~branch['records'][start+t]['done']
        if t in (0,1,3,7):
            real=rotation(d['plus']['records'][start+t]['next_physical'],d['minus']['records'][start+t]['next_physical'])
            model=rotation(pred['plus'],pred['minus'])
            resolved=mask.clone()
            noise=torch.zeros(len(ctx))
            if replicated:
                physical={k:v['records'][start+t]['next_physical'] for k,v in d.items()}
                for k in ('repeat','repeat2','repeat3'):
                    noise=torch.maximum(noise,rotation(physical[k],physical['base']).norm(dim=-1))
                for a,b in (('plus','plus2'),('minus','minus2')):
                    noise=torch.maximum(noise,rotation(physical[a],physical[b]).norm(dim=-1))
                real=(real+rotation(physical['plus2'],physical['minus2']))/2
                resolved &= real.norm(dim=-1)>10*noise.clamp_min(1e-6)
                for k in d: resolved &= ~d[k]['records'][start+t]['done']
            report['horizons'][str(t+1)]=stats(real,model,resolved)
            report['horizons'][str(t+1)]['eligible_samples']=int(mask.sum())
            report['horizons'][str(t+1)]['repeat_noise_deg_max']=noise.max().item()*180/math.pi
            report['horizons'][str(t+1)]['repeat_noise_deg_median']=noise.median().item()*180/math.pi
            real_reward=d['plus']['records'][start+t]['reward']-d['minus']['records'][start+t]['reward']
            if replicated:
                real_reward=(real_reward+d['plus2']['records'][start+t]['reward']-d['minus2']['records'][start+t]['reward'])/2
            def model_reward(k):
                return reward(before[k],pred[k],d[k]['records'][start+t]['action'],cfg.control_dt,cfg.success_bonus,cfg.action_rate_weight,cfg.joint_vel_weight,cfg.action_l2_weight,cfg.action_second_diff_weight,hold_cfg=cfg)
            model_reward_delta=model_reward('plus')-model_reward('minus')
            reward_noise=torch.zeros_like(real_reward)
            if replicated:
                for a,b in [('repeat','base'),('repeat2','base'),('repeat3','base'),('plus','plus2'),('minus','minus2')]:
                    reward_noise=torch.maximum(reward_noise,(d[a]['records'][start+t]['reward']-d[b]['records'][start+t]['reward']).abs())
            reward_mask=resolved & (real_reward.abs()>10*reward_noise.clamp_min(1e-9))
            report['horizons'][str(t+1)]['reward_sign_samples']=int(reward_mask.sum())
            report['horizons'][str(t+1)]['reward_sign_agreement']=(real_reward[reward_mask]*model_reward_delta[reward_mask]>0).float().mean().item() if reward_mask.any() else None
            report['horizons'][str(t+1)]['base_orientation_error_deg']=rotation(pred['base'],d['base']['records'][start+t]['next_physical'])[mask].norm(dim=-1).mean().item()*180/math.pi
(folder/'response.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
