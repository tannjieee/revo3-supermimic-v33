"""Bound RSLSphere65 Newton preview and short diagnostics. No policy/checkpoint loaded."""
import argparse, json, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.utils.runtime import SimulationRuntime,add_runtime_args,parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID,TransitionRecorderCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--env_config',type=Path,default=ROOT/'evidence/env_sphere20.yaml')
p.add_argument('--output_dir',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=1)
p.add_argument('--steps',type=int,default=440)
p.add_argument('--seed',type=int,default=42)
add_runtime_args(p,TASK_ID);args=parse_runtime_args(p)
binding=SavedEnvironmentBinding(args.env_config)
def configure(cfg):
 binding.configure(cfg,args);configure_clean_state_observations(cfg)
 cfg.recorders=TransitionRecorderCfg();cfg.scene.clone_in_fabric=False
 cfg.scene.robot.spawn.activate_contact_sensors=False;cfg.scene.object.spawn.activate_contact_sensors=False
 cfg.commands.object_pose.debug_vis=False;cfg.seed=args.seed
 if 'newton_gl' in (args.visualizer or []):
  v=NewtonGLVisualizerCfg(headless=True,window_width=1280,window_height=960,
   eye=(-.22,-.24,.78),lookat=(.02,0,.565),streaming_view=False,enable_picking=False,background_color=(.08,.11,.16))
  cfg.sim.visualizer_cfgs=[v];cfg.sim.default_visualizer_cfg=v
 binding.verify(cfg)
app=SimulationRuntime(args,configure=configure)
import gymnasium as gym
import torch,numpy as np
from PIL import Image
from importlib.metadata import version
from isaaclab_newton.physics import NewtonManager
from isaaclab.utils.io import dump_yaml
import viserdex
from viserdex.supermimic.simulation import Adapter,tensor,collision_shape_mask
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import *

def main():
 args.output_dir.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(args.seed)
 env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
 result=dict(passed=False,checkpoint=None,controller='hold initial joints then diagnostic actions; no learned policy',
  source_module=viserdex.__file__,geometry_module=sys.modules['viserdex.supermimic.geometry'].__file__,
  versions={k:version(k) for k in ('isaacsim','isaaclab','newton','warp-lang')},binding=binding.verify(env.cfg))
 try:
  assert Path(viserdex.__file__).is_relative_to(ROOT)
  clean,_=env.reset();a=Adapter(env,PerceptionNoiseCfg());a.audit_observation(clean)
  assert a.geometry.object_shape=='sphere' and a.geometry.feature_dim==300 and a.history.shape[1:]==(10,124)
  result.update(geometry=a.geometry_spec,geometry_digest=a.geometry.digest,context_schema=a.context_schema,
    object_mass=tensor(a.obj.data.body_mass).tolist(),object_inertia=tensor(a.obj.data.body_inertia).tolist())
  model=env.sim.physics_manager.get_model();mask=collision_shape_mask(a.obj,model,env.num_envs)
  result['live_shape_type']=tensor(a.obj._root_view.get_attribute('shape_type',model)).reshape(env.num_envs,-1)[mask].tolist()
  result['live_shape_scale']=tensor(a.obj._root_view.get_attribute('shape_scale',model)).reshape(env.num_envs,-1,3)[mask].tolist()
  # Analytic checks include inside/outside/center and an independent nearest point.
  x=torch.tensor([[.0425,0,0],[0,.0225,0],[0,0,-.0325],[0,0,0]],device=env.device,requires_grad=True)
  gap,n=a.geometry.surface(x)
  torch.testing.assert_close(gap[:,0],x.new_tensor([.01,-.01,0,-.0325]),atol=1e-7,rtol=0)
  torch.testing.assert_close(n.norm(dim=-1),torch.ones(4,device=env.device))
  nearest=x-gap*n
  torch.testing.assert_close(nearest.norm(dim=-1),torch.full((4,),.0325,device=env.device),atol=1e-7,rtol=0)
  (gap.square().sum()+n.square().sum()).backward();assert torch.isfinite(x.grad).all()
  # Full features and geometric supervision remain differentiable in q, pose and twist.
  s=a.state.detach().clone().requires_grad_(True)
  loss=a.geometry(s).square().mean()+a.geometry.supervised_errors(s,a.state.detach()).mean()
  loss.backward();assert torch.isfinite(s.grad).all()
  result['analytic_geometry_and_gradient']='passed'
  lo,hi=a.limits.unbind(-1);hold=(2*(a.state[:,Q]-lo)/(hi-lo)-1).clamp(-1,1)
  viz=next((v for v in env.sim.visualizers if v.cfg.visualizer_type=='newton_gl'),None)
  if viz:
   origin=env.scene.env_origins[0].cpu().numpy()
   env.sim.set_camera_view(tuple(origin+[-.22,-.24,.78]),tuple(origin+[.02,0,.565]))
  data=NewtonManager._solver.mjw_data
  result['contact_fields']=[v for v in dir(data.contact) if not v.startswith('_')]
  peaks=dict(candidates=0,contacts=0,constraints=0);rows=[]
  for step in range(args.steps):
   if step<160: action=hold
   elif step<240: action=torch.zeros_like(hold)
   elif step<320: action=torch.empty_like(hold).uniform_(-1,1)
   else: action=hold
   row,_,_,clean=a.step(action)
   assert torch.isfinite(a.state).all() and torch.isfinite(row['reward']).all() and torch.isfinite(a.geometry(a.state)).all()
   overflow=int(np.bitwise_or.reduce(data.overflow.numpy().ravel()));assert overflow==0
   peaks['candidates']=max(peaks['candidates'],int(data.ncollision.numpy().max()))
   peaks['contacts']=max(peaks['contacts'],int(data.nacon.numpy().max()))
   peaks['constraints']=max(peaks['constraints'],int(data.nefc.numpy().max()))
   if step in (0,1,4,19,39,79,159,239,319,args.steps-1):
    entry=dict(step=step+1,pos=tensor(a.obj.data.root_pos_w).tolist(),vel=tensor(a.obj.data.root_lin_vel_w).tolist(),resets=a.reset_count)
    if hasattr(data.contact,'dist'):
     dist=data.contact.dist.numpy();entry['contact_dist_min']=float(dist.min())
    rows.append(entry)
    if viz:
     for _ in range(3):env.sim.render()
     pixels=viz.render_rgb_array()[...,:3]
     assert pixels.std()>5
     Image.fromarray(pixels).save(args.output_dir/f'sphere_step_{step+1:04d}.png')
   if step%100==0:print('PREVIEW_STEP',step,flush=True)
  # Verify timeout path explicitly without modifying configured episode/reward logic.
  env.episode_length_buf[:]=env.max_episode_length-1;a.state=a.read_state()
  row,_,_,_=a.step(hold);assert row['done'].all()
  result.update(passed=True,steps=args.steps+1,control_hz=1/env.step_dt,episode_seconds=env.max_episode_length*env.step_dt,
    parity_max_errors=a.max_errors,resets=a.reset_count,contact_peaks=peaks,trajectory=rows,
    success_threshold_rad=.1,forced_timeout_verified=True,policy_trained=False)
  dump_yaml(str(args.output_dir/'env.yaml'),env.cfg)
  print('SPHERE_PREVIEW_PASSED',json.dumps(result,default=str),flush=True)
 finally:
  (args.output_dir/'validation.json').write_text(json.dumps(result,indent=2,default=str))
  env.close();app.close()
if __name__=='__main__':main()
