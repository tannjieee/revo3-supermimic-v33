"""Read-only scale equivalence and real-vs-model branch objective audit.
No short-horizon return is presented as a long-horizon value ground truth.
"""
import copy,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/viserdex'))
from viserdex.supermimic.core import Config,action_target,advance,reward
from viserdex.supermimic.networks import Dynamics,Critic
from viserdex.supermimic.constraints import violation_costs
from viserdex.supermimic.stable_values import failure_risk_cost
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.stability_reward import task_reward_bound
from viserdex.supermimic.spatial_contract import drop_distance

def main():
    checkpoint=Path(sys.argv[1]);root=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);p=torch.load(checkpoint,weights_only=False,map_location='cpu');cfg=Config(**p['config'])
    wm=Dynamics(65,cfg).eval();wm.load_state_dict(p['wm']);critic=Critic(65,cfg).eval();critic.load_state_dict(p['target_critic'])
    report={'checkpoint':str(checkpoint),'gamma_to_horizon':cfg.gamma**8,'scope':'Frozen open-loop branch objectives; finite real prefixes do NOT identify true tail values.'}
    folder=root/'replicated_snapshot';raw=torch.load(folder/'base.pt',weights_only=False);s=raw['records'][0]['state'];ctx=raw['context'];limits=raw['limits'];a=raw['records'][0]['action']
    report['scaling']={}
    for name,network in [('wm',wm),('critic',critic)]:
        m=copy.deepcopy(network);sl=slice(51,54);oldstd=m.input_norm.std.clone()
        scale=m.input_scale[sl];stats=dict(physical_std_equivalent=(oldstd[sl]/scale).tolist(),gain_per_m_s=(scale/oldstd[sl]).tolist())
        with torch.no_grad():
            m.input_scale[sl]/=2;m.input_norm.mean[sl]/=2;m.input_norm.variance[sl]/=4
            # Reparameterize first-layer weights to preserve the exact function.
            m.net[0].weight[:,sl]*=(2*m.input_norm.std[sl]/oldstd[sl])[None]
        def call(net,state):return net(state,action_target(state,a,limits,cfg),ctx) if name=='wm' else net(state,ctx)
        x=s.clone().requires_grad_(True);original=call(network,x);new=call(m,x)
        g1=torch.autograd.grad(original.sum(),x,retain_graph=True)[0];g2=torch.autograd.grad(new.sum(),x)[0]
        stats['reparameterized_output_max_abs_error']=(new-original).abs().max().item();stats['input_gradient_max_abs_error']=(g1-g2).abs().max().item();stats['input_gradient_relative_norm_error']=((g1-g2).norm()/g1.norm().clamp_min(1e-20)).item()
        stats['scale_2_to_1_without_weight_compensation_gain_ratio']=(oldstd[sl]/(2*m.input_norm.std[sl])).tolist()
        report['scaling'][name]=stats
    def objectives(data,model=False):
        context,lim=data['context'],data['limits']; rows=data['records']; state=rows[0]['state'];alive=torch.ones(len(state));rout=torch.zeros_like(alive);cost=torch.zeros_like(alive)
        for t,row in enumerate(rows):
            act=row['action'];target=action_target(state,act,lim,cfg)
            nxt=wm(state,target,context) if model else row['next_physical']
            r=reward(state,nxt,act,cfg.control_dt,cfg.success_bonus,cfg.action_rate_weight,cfg.joint_vel_weight,cfg.action_l2_weight,cfg.action_second_diff_weight,hold_cfg=cfg) if model else row['reward']
            prob=wm.termination_probability(nxt) if model else row['terminated'].float()
            if model:prob=torch.where(drop_distance(nxt,cfg)>cfg.drop_radius_m,1.,prob)
            raw,_=violation_costs(state,nxt,act,lim,cfg)
            rout+=cfg.gamma**t*alive*r
            cost+=cfg.gamma**t*alive*failure_risk_cost(raw,prob,cfg).sum(-1)
            alive*=1-prob
            advanced=advance(state,nxt,act,target,state[:,COMPACT.goal_rot],500,hold_cfg=cfg)
            state=torch.where((alive>0)[:,None],advanced,state)
        val=critic(state,context)
        return {'reward':-(1-cfg.gamma)*rout/task_reward_bound(cfg),'cost':cfg.cost_penalty*(1-cfg.gamma)*cost,
            'tail_reward':-(1-cfg.gamma)*cfg.gamma**8*alive*val[:,0]/task_reward_bound(cfg),
            'tail_cost':cfg.cost_penalty*(1-cfg.gamma)*cfg.gamma**8*alive*val[:,1:].sum(-1)}
    report['branches']={}
    with torch.no_grad():
        for folder in [root/'replicated_snapshot',root/'replicated_snapshot_seed45']:
            data={k:torch.load(folder/(k+'.pt'),weights_only=False) for k in ('base','repeat','repeat2','repeat3','plus','minus','plus2','minus2')}
            real={k:objectives(v) for k,v in data.items()};model={k:objectives(data[k],True) for k in ('plus','minus')}
            valid=torch.ones(64,dtype=torch.bool)
            for v in data.values():valid &= ~torch.stack([r['done'] for r in v['records']]).any(0)
            result={}
            for metric,fields in [('short_reward',['reward']),('short_constrained',['reward','cost']),('critic_augmented',['reward','cost','tail_reward','tail_cost'])]:
                values={k:sum(v[f] for f in fields) for k,v in real.items()}
                diff=((values['plus']-values['minus'])+(values['plus2']-values['minus2']))/2
                noise=torch.zeros_like(diff)
                for aa,bb in [('repeat','base'),('repeat2','base'),('repeat3','base'),('plus','plus2'),('minus','minus2')]:noise=torch.maximum(noise,(values[aa]-values[bb]).abs())
                resolved=valid & (diff.abs()>10*noise.clamp_min(1e-10))
                tests={}
                for weight in (0.,.1,1.):
                    md=sum(model['plus'][f]-model['minus'][f] for f in ('reward','cost'))+weight*sum(model['plus'][f]-model['minus'][f] for f in ('tail_reward','tail_cost'))
                    tests[str(weight)]={'agreement':((md[resolved]*diff[resolved])>0).float().mean().item() if resolved.any() else None,'relative_error':((md[resolved]-diff[resolved]).norm()/diff[resolved].norm()).item() if resolved.any() else None}
                result[metric]={'resolved_samples':int(resolved.sum()),'tail_weights':tests}
            report['branches'][folder.name]=result
    (out/'scale_bootstrap.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
