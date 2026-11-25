"""Frozen-model fresh-data rollout diagnostic; never update policy or world model."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/viserdex"))
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID, TransitionRecorderCfg
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--env_config", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--steps", type=int, default=640)
p.add_argument("--seed", type=int, default=43)
p.add_argument("--exploration", type=float, default=0.1)
add_runtime_args(p, TASK_ID)
args = parse_runtime_args(p)
binding = SavedEnvironmentBinding(args.env_config)
def configure(cfg):
    binding.configure(cfg,args)
    configure_clean_state_observations(cfg)
    cfg.recorders=TransitionRecorderCfg()
    cfg.scene.clone_in_fabric=False
    cfg.scene.robot.spawn.activate_contact_sensors=False
    cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False
    cfg.seed=args.seed
    binding.verify(cfg)
app=SimulationRuntime(args,configure=configure)
import torch
import gymnasium as gym
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import Config, physical_delta, ema_target, reward
from viserdex.supermimic.networks import Actor, Dynamics
from viserdex.supermimic.simulation import Adapter

payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
assert payload["version"] == 13, "Legacy checkpoints need their original source"
cfg=Config(**payload["config"])
torch.manual_seed(args.seed)
env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
try:
    obs,_=env.reset()
    adapter=Adapter(env,PerceptionNoiseCfg(**payload["noise_config"]),cfg.history)
    actor=Actor(cfg).to(env.device).eval(); actor.load_state_dict(payload["actor"])
    wm=Dynamics(adapter.context.shape[-1],cfg,adapter.geometry_spec).to(env.device).eval();wm.load_state_dict(payload["wm"])
    pending=[]; aggregates={};goals=episodes=drops=saturated=total_actions=0
    episode_goals=torch.zeros(env.num_envs,device=env.device)
    completed_goals=0
    with torch.no_grad():
        for step in range(args.steps):
            action=actor.sample(adapter.history,args.exploration)
            saturated+=int((action.abs()>1).sum());total_actions+=action.numel()
            row,_,_,_=adapter.step(action)
            reached=(adapter.captured_orientation_error<.1)&~row["done"]
            goals+=int(reached.sum());episode_goals+=reached
            episodes+=int(row["done"].sum());completed_goals+=float(episode_goals[row["done"]].sum());episode_goals[row["done"]]=0
            drops+=int(((row["next_physical"][:,42:45].norm(dim=-1)>.3)&row["terminated"]).sum())
            pending.append(row)
            if len(pending)==32:
                state=pending[0]["state"].clone();valid=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
                for t,real in enumerate(pending,1):
                    target=action_target(state,real["action"],adapter.limits,cfg)
                    nxt=wm(state,target,adapter.context)
                    if t in (1,8,16,32):
                        error=physical_delta(real["next_physical"],nxt)
                        pred_drop=nxt[:,42:45].norm(dim=-1)>.3
                        true_drop=(real["next_physical"][:,42:45].norm(dim=-1)>.3)&real["terminated"]
                        a=aggregates.setdefault(t,dict(n=0,pos_sq=0.,rot_sq=0.,true_drop=0,pred_drop=0,hit_drop=0,real_reward=0.,pred_reward=0.))
                        a["n"]+=int(valid.sum());a["pos_sq"]+=float(error[valid,42:45].square().sum());a["rot_sq"]+=float(error[valid,45:48].square().sum())
                        a["true_drop"]+=int((true_drop&valid).sum());a["pred_drop"]+=int((pred_drop&valid).sum());a["hit_drop"]+=int((pred_drop&true_drop&valid).sum())
                        a["real_reward"]+=float(real["reward"][valid].sum())
                        with_real_goal=torch.cat((state[:,:76],real["state"][:,76:]),-1)
                        a["pred_reward"]+=float(reward(with_real_goal,nxt,real["action"],cfg.control_dt,cfg.success_bonus,cfg.action_rate_weight)[valid].sum())
                    valid &= ~real["done"]
                    state=torch.cat((nxt,target,state[:,76:]),-1)
                pending=[]
                print("steps",step+1,flush=True)
    for a in aggregates.values():
        a["position_component_rmse_m"]=(a["pos_sq"]/(3*a["n"]))**.5
        a["rotation_component_rmse_rad"]=(a["rot_sq"]/(3*a["n"]))**.5
        a["pred_reward_mean"]=a["pred_reward"]/a["n"];a["real_reward_mean"]=a["real_reward"]/a["n"]
    result=dict(iteration=payload["iteration"],seed=args.seed,num_envs=env.num_envs,steps=args.steps,exploration=args.exploration,episodes=episodes,completed_goals=completed_goals,goals_per_completed_episode=completed_goals/max(1,episodes),drops=drops,drop_fraction=drops/max(1,episodes),action_saturation=saturated/total_actions,windows=aggregates,max_errors=adapter.max_errors)
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:
    env.close();app.close()
