"""Frozen real-trajectory value consistency and input coverage; no fit."""
import sys,json,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/viserdex'))
from viserdex.supermimic.core import Config, action_target, advance, rotation_log, quat_mul, quat_conjugate
from viserdex.supermimic.networks import Actor, Critic, Dynamics
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.stable_values import failure_risk_cost
from viserdex.supermimic.constraints import violation_costs

torch.set_num_threads(4)
data=torch.load(sys.argv[1],weights_only=False,map_location='cpu'); out=Path(sys.argv[2])
p=torch.load(data['checkpoint'],weights_only=False,map_location='cpu'); cfg=Config(**p['config'])
actor=Actor(cfg).eval();actor.load_state_dict(p['actor'])
critic=Critic(data['context'].shape[-1],cfg).eval();critic.load_state_dict(p['target_critic'])
wm=Dynamics(data['context'].shape[-1],cfg).eval();wm.load_state_dict(p['wm'])
rows=data['records']; ctx=data['context']; limits=data['limits']; half=(limits[:,:,1]-limits[:,:,0])/2
report={'checkpoint':data['checkpoint'],'scope':'Frozen deterministic evaluation; not behavior replay, cannot measure training off-policy bias directly.'}
with torch.no_grad():
    report['value_consistency']={}
    for horizon in (1,8,32,128):
        values=[]; labels=[]; bootparts=[]; immediateparts=[]
        for start in range(40,len(rows)-horizon,32):
            state=rows[start]['state']; alive=torch.ones(len(ctx),dtype=torch.bool)
            target=torch.zeros(len(ctx),6); boot=torch.zeros_like(target)
            for offset in range(horizon):
                row=rows[start+offset]; raw,_=violation_costs(row['state'],row['next_physical'],row['action'],limits,cfg)
                term=row['terminated'].bool(); done=row['done'].bool()
                immediate=torch.cat((row['reward'][:,None],failure_risk_cost(raw,term,cfg)),dim=-1)
                target+=cfg.gamma**offset*alive[:,None]*immediate
                nxt=advance(row['state'],row['next_physical'],row['action'],action_target(row['state'],row['action'],limits,cfg),row['state'][:,COMPACT.goal_rot],data['episode_steps'],hold_cfg=cfg)
                boundary=alive & ~term & (done | (offset==horizon-1))
                boot+=cfg.gamma**(offset+1)*boundary[:,None]*critic(nxt,ctx)
                alive &= ~done
            values.append(critic(state,ctx)); labels.append(target+boot);bootparts.append(boot);immediateparts.append(target)
        v,y,b,i=map(torch.cat,(values,labels,bootparts,immediateparts));bounds=critic.value_bounds
        report['value_consistency'][str(horizon)]={'samples':len(v),'value_mean':v.mean(0).tolist(),'target_mean':y.mean(0).tolist(),'normalized_mae':((v-y).abs()/bounds).mean(0).tolist(),'normalized_signed_overestimate':((v-y)/bounds).mean(0).tolist(),'reward_bootstrap_fraction':(b[:,0].sum()/y[:,0].sum()).item()}
    s=torch.cat([r['state'] for r in rows[40:]]); h=torch.cat([r['history'] for r in rows[40:]])
    a=torch.cat([r['action'] for r in rows[40:]]);nxt=torch.cat([r['next_physical'] for r in rows[40:]])
    rot=rotation_log(quat_mul(nxt[:,45:49],quat_conjugate(s[:,45:49]))).norm(dim=-1)*180/math.pi
    err=rotation_log(quat_mul(s[:,45:49],quat_conjugate(s[:,COMPACT.goal_rot]))).norm(dim=-1)*180/math.pi
    report['coverage']={'orientation_increment_deg_quantiles':torch.quantile(rot,torch.tensor([.1,.5,.9,.99])).tolist(),'angle_under_30deg_fraction':(err<30).float().mean().item(),'angle_under_10deg_fraction':(err<10).float().mean().item(),'wm_rotation_delta_std_deg':(wm.delta_norm.std[45:48]*180/math.pi).tolist(),'actor_goal_input_std':actor.obs_norm.std[-6:].tolist(),'per_joint_target_std_deg':(a.reshape(-1,len(ctx),21)*half*180/math.pi).flatten(0,1).std(0).tolist()}
    valid=~torch.stack([r['done'] for r in rows[:250]]).any(0)
    goal=rows[0]['state'][:,COMPACT.goal_rot]
    def error(q):return rotation_log(quat_mul(q,quat_conjugate(goal))).norm(dim=-1)*180/math.pi
    improvement=error(rows[0]['state'][:,45:49])-error(rows[249]['next_physical'][:,45:49])
    report['paired_ten_second_progress']={'surviving_worlds':int(valid.sum()),'mean_improvement_deg':improvement[valid].mean().item(),'median_improvement_deg':improvement[valid].median().item(),'improved_10deg':int(((improvement>10)&valid).sum()),'worsened_10deg':int(((improvement< -10)&valid).sum())}
(out/'values.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
