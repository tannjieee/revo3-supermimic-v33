"""Read-only frozen-policy probes on independently collected Newton transitions."""
from pathlib import Path
import json, sys, math, types
import torch

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO/'source/viserdex'))
from viserdex.supermimic.core import Config, action_target, advance, reward, push_history
from viserdex.supermimic.networks import Actor, Dynamics, Critic
from viserdex.supermimic.trainer import Trainer
from viserdex.supermimic.stable_values import running_risk, imagine_stable
from viserdex.supermimic.constraints import violation_costs, cost_names
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.sensors import observe
from viserdex.supermimic.stability_reward import task_reward_bound
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.policy_contract import rotation_6d_xyzw

torch.set_num_threads(4)
torch.manual_seed(44048)
data = torch.load(Path(sys.argv[1]), map_location='cpu', weights_only=False)
payload = torch.load(data['checkpoint'], map_location='cpu', weights_only=False)
cfg = Config(**payload['config'])
rows = data['records']
device = 'cuda:0'
t = types.SimpleNamespace(cfg=cfg, noise_cfg=PerceptionNoiseCfg(**payload['noise_config']),
    context=data['context'].to(device), limits=data['limits'].to(device),
    episode_steps=data['episode_steps'])
t.actor = Actor(cfg).to(device)
t.wm = Dynamics(t.context.shape[-1], cfg).to(device)
t.target_critic = Critic(t.context.shape[-1], cfg).to(device)
for name in ('actor', 'wm', 'target_critic'):
    getattr(t,name).load_state_dict(payload[name]); getattr(t,name).eval()
t.wm.requires_grad_(False); t.target_critic.requires_grad_(False)
t.continuation = types.MethodType(Trainer.continuation, t)
names = cost_names(cfg)
result = dict(checkpoint=data['checkpoint'], seed=44048,
              scope='64 fresh evaluation worlds, seed 43; no optimizer or normalization updates; not the live training replay batch',
              cost_names=names)


def batch_for(seed, size=128):
    gen=torch.Generator().manual_seed(seed)
    steps=torch.randint(40,len(rows),(size,),generator=gen)
    envs=torch.randint(len(data['context']),(size,),generator=gen)
    b={k:torch.stack([rows[s][k][e] for s,e in zip(steps,envs)]).to(device) for k in rows[0]}
    b['env_id']=envs.to(device)
    return b


def decomposed(b, detach_bootstrap=None, detach_survival=False):
    state, history=b['state'],b['history']
    sensors={k[7:]:v for k,v in b.items() if k.startswith('sensor/')}
    context, limits=t.context[b['env_id']],t.limits[b['env_id']]
    alive=torch.ones(len(state),device=device)
    objective=torch.zeros_like(alive)
    running=torch.zeros(len(state),len(names),device=device);terminal=torch.zeros_like(running)
    actions=[];logits=[];raws=[]
    from viserdex.supermimic.absolute_policy import explore_absolute, logit_regularizer
    noise=b['exploration_noise_rad']
    for step in range(cfg.horizon):
        active=alive>0
        z=t.actor.logits(history); action,next_noise=explore_absolute(z.tanh(),noise,limits,cfg)
        if step==0: regularization=logit_regularizer(z,cfg).mean()
        actions.append(action);logits.append(z)
        target=action_target(state,action,limits,cfg)
        nxt=t.wm(state,target,context)
        r=reward(state,nxt,action,cfg.control_dt,cfg.success_bonus,cfg.action_rate_weight,
                 cfg.joint_vel_weight,cfg.action_l2_weight,cfg.action_second_diff_weight,hold_cfg=cfg)
        raw,_=violation_costs(state,nxt,action,limits,cfg);raws.append(raw.detach())
        continuation=torch.where(active,t.continuation(state,nxt),0.)
        p=1-continuation;discount_alive=cfg.gamma**step*alive
        objective=objective+discount_alive*r
        running=running+discount_alive[:,None]*continuation[:,None]*running_risk(raw)
        terminal=terminal+discount_alive[:,None]*p[:,None]/(1-cfg.gamma)
        alive=alive*continuation
        advanced=advance(state,nxt,action,target,state[:,COMPACT.goal_rot],t.episode_steps,hold_cfg=cfg)
        survives=alive>0
        noise=torch.where(survives[:,None],next_noise,noise)
        state=torch.where(survives[:,None],advanced,state)
        obs,next_sensors=observe(state,sensors,limits,t.noise_cfg,cfg)
        history=torch.where(survives[:,None,None],push_history(history,obs),history)
        sensors={k:torch.where(survives.reshape(-1,*([1]*(v.ndim-1))),v,sensors[k]) for k,v in next_sensors.items()}
        value_state=state
        if detach_bootstrap is not None:
            value_state=state.clone()
            sl=COMPACT.last_action if detach_bootstrap=='action' else slice(0,55)
            value_state[:,sl]=value_state[:,sl].detach()
        value=t.target_critic(value_state,context)
    bootstrap_alive=alive.detach() if detach_survival else alive
    parts={'negative_reward_rollout':-(1-cfg.gamma)*objective.mean()/task_reward_bound(cfg),
           'negative_reward_bootstrap':-(1-cfg.gamma)*(cfg.gamma**cfg.horizon*bootstrap_alive*value[:,0]).mean()/task_reward_bound(cfg)}
    for label,v in [('running',(1-cfg.gamma)*running.mean(0)),('terminal',(1-cfg.gamma)*terminal.mean(0)),
                    ('bootstrap',(1-cfg.gamma)*(cfg.gamma**cfg.horizon*bootstrap_alive[:,None]*value[:,1:]).mean(0))]:
        for i,name in enumerate(names): parts[label+'_'+name]=cfg.cost_penalty*v[i]
    parts['regularizer']=regularization
    return parts,actions,logits,torch.stack(raws)


def gradient_probe(seed):
    b=batch_for(seed)
    torch.manual_seed(seed)
    parts,actions,logits,raws=decomposed(b)
    total=sum(parts.values())
    torch.manual_seed(seed)
    with torch.no_grad(): reference=imagine_stable(t,b)[0]
    assert abs(total.item()-reference.item())<2e-6,(total,reference)
    grouped={'reward_rollout':parts['negative_reward_rollout'], 'reward_bootstrap':parts['negative_reward_bootstrap']}
    for kind in ('running','bootstrap'):
        for name in names: grouped[kind+'_'+name]=parts[kind+'_'+name]
    grouped['terminal_all']=sum(parts['terminal_'+name] for name in names)
    grouped['reward_total']=grouped['reward_rollout']+grouped['reward_bootstrap']
    grouped['regularizer']=parts['regularizer']
    grouped['cost_total']=total-grouped['reward_total']-parts['regularizer']
    grouped['total']=total
    params=tuple(t.actor.parameters()); vectors={};details={}
    for name,loss in grouped.items():
        gs=torch.autograd.grad(loss,params+(actions[0],logits[0]),retain_graph=True,allow_unused=True)
        vector=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])
        vectors[name]=vector.detach()
        details[name]=dict(value=loss.item(),parameter_gradient_norm=vector.norm().item(),
            first_action_gradient_norm=0 if gs[-2] is None else gs[-2].norm().item(),
            first_logit_gradient_norm=0 if gs[-1] is None else gs[-1].norm().item())
    for name, v in vectors.items():
        details[name]['cosine_with_total_gradient']=(v@vectors['total']/(v.norm()*vectors['total'].norm()).clamp_min(1e-20)).item()
        details[name]['projection_fraction_of_total']=(v@vectors['total']/vectors['total'].square().sum().clamp_min(1e-20)).item()
    reward_cost_cos=(vectors['reward_total']@vectors['cost_total']/(vectors['reward_total'].norm()*vectors['cost_total'].norm()).clamp_min(1e-20)).item()
    return dict(seed=seed,batch_size=len(b['state']),exact_objective_error=abs(total.item()-reference.item()),
                components=details,cost_to_reward_parameter_gradient_ratio=(vectors['cost_total'].norm()/vectors['reward_total'].norm()).item(),
                reward_cost_gradient_cosine=reward_cost_cos,
                max_running_raw_cost=raws.amax((0,1)).tolist())



result['gradients']=[gradient_probe(seed) for seed in (41,42,43)]
result['bootstrap_path_scope']='Survival weight held constant only for path decomposition; main gradient probes use unchanged full objective.'
result['bootstrap_paths']=[]
for seed in (41,42,43):
    b=batch_for(seed); vectors={}
    for detach in (None,'action','physical'):
        torch.manual_seed(seed);parts,_,_,_=decomposed(b,detach,detach_survival=True)
        loss=sum(v for k,v in parts.items() if 'bootstrap' in k)
        gs=torch.autograd.grad(loss,tuple(t.actor.parameters()))
        vectors[str(detach)]=torch.cat([g.flatten() for g in gs]).detach()
    full=vectors['None']; physical=vectors['action']; direct=vectors['physical']
    result['bootstrap_paths'].append(dict(seed=seed,full_norm=full.norm().item(),
        physical_path_norm=physical.norm().item(),direct_last_action_path_norm=direct.norm().item(),
        direct_projection_fraction=(direct@full/full.square().sum()).item(),
        physical_projection_fraction=(physical@full/full.square().sum()).item(),
        additivity_relative_error=(full-direct-physical).norm().item()/full.norm().item()))
(OUT/'gradient.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
