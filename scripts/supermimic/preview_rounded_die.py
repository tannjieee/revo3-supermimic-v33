"""Preview real Newton loading/contact with fixed joint targets; no policy or optimizer."""
import argparse, copy, json, sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.utils.runtime import SimulationRuntime,add_runtime_args,parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID,TransitionRecorderCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--env_config',type=Path,default=ROOT/'configs/rounded_die_wuji_handpose/env.yaml')
p.add_argument('--output_dir',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=1)
p.add_argument('--steps',type=int,default=200)
p.add_argument('--seed',type=int,default=42)
add_runtime_args(p,TASK_ID);args=parse_runtime_args(p)
binding=SavedEnvironmentBinding(args.env_config)
def configure(cfg):
    binding.configure(cfg,args);configure_clean_state_observations(cfg)
    cfg.recorders=TransitionRecorderCfg();cfg.scene.clone_in_fabric=False
    cfg.scene.robot.spawn.activate_contact_sensors=False;cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False;cfg.seed=args.seed
    if 'newton_gl' in (args.visualizer or []):
        assert args.num_envs==1
        v=NewtonGLVisualizerCfg(headless=True,window_width=1280,window_height=960,
            eye=(-.22,-.24,.78),lookat=(.02,0,.565),streaming_view=False,enable_picking=False,
            background_color=(.08,.11,.16))
        cfg.sim.visualizer_cfgs=[v];cfg.sim.default_visualizer_cfg=v
    binding.verify(cfg)
app=SimulationRuntime(args,configure=configure)
import gymnasium as gym
import torch,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from isaaclab_newton.physics import NewtonManager
from isaaclab.utils.io import dump_yaml
from isaaclab.markers import VisualizationMarkers,VisualizationMarkersCfg
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.utils.math import quat_error_magnitude
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.simulation import Adapter,tensor
from viserdex.supermimic.core import Q


def main():
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
    writer=None
    result=dict(task_id=TASK_ID,passed=False,training_started=False,network_updates=0,
        policy_loaded=False,controller='fixed reset joint targets through implicit PD and EMA=0.5',
        seed=args.seed,num_envs=args.num_envs,binding=binding.verify(env.cfg))
    try:
        clean,_=env.reset();noise=PerceptionNoiseCfg();a=Adapter(env,noise);a.audit_observation(clean)
        assert a.history.shape[1:]==(10,124) and a.geometry.feature_dim==300
        result.update(noise_config=asdict(noise),geometry=a.geometry_spec,live_geometry=a.convex_audit,
            mass_kg=tensor(a.obj.data.body_mass).tolist(),inertia_kg_m2=tensor(a.obj.data.body_inertia).tolist())
        state=a.state.detach().clone().requires_grad_(True)
        a.geometry(state).square().mean().backward()
        assert torch.isfinite(state.grad).all()
        result['geometry_gradient_finite']=True
        lo,hi=a.limits.unbind(-1)
        def held_action():return torch.zeros_like(a.state[:,Q])
        hold=held_action()
        viz=next((v for v in env.sim.visualizers if v.cfg.visualizer_type=='newton_gl'),None)
        marker=None
        if viz:
            from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Sdf
            asset=Path(env.cfg.scene.object.spawn.usd_path)
            stage=Usd.Stage.Open(str(asset));layer=stage.Flatten()
            UsdUtils.ModifyAssetPaths(layer,lambda s:str((asset.parent/s).resolve()) if s and not s.startswith('/') else s)
            stage=Usd.Stage.Open(layer)
            stage.RemovePrim('/Object/Collision')
            root=stage.GetDefaultPrim()
            root.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit([]))
            for name in list(root.GetPropertyNames()):
                if name.startswith(('physics:','physxRigidBody:')):root.RemoveProperty(name)
            target_path=args.output_dir/'target_visual.usda';stage.GetRootLayer().Export(str(target_path.resolve()))
            proto=copy.deepcopy(env.cfg.commands.object_pose.goal_pose_visualizer_cfg.markers['goal'])
            proto.usd_path=str(target_path.resolve());proto.scale=(1.,1.,1.)
            proto.visual_material=PreviewSurfaceCfg(diffuse_color=(1.,1.,1.))
            marker=VisualizationMarkers(VisualizationMarkersCfg(prim_path='/Visuals/DiePreviewTarget',markers={'die':proto}))
            command=env.command_manager.get_term('object_pose')
            marker_pos=command.pos_command_w[:1]+torch.tensor([[.14,-.14,.05]],device=env.device)
            origin=env.scene.env_origins[0].cpu().numpy()
            env.sim.set_camera_view(tuple(origin+[-.22,-.24,.78]),tuple(origin+[.02,0,.565]))
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
            small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
            def frame(t):
                marker.visualize(translations=marker_pos,orientations=command.quat_command_w[:1])
                env.sim.render()
                image=Image.fromarray(viz.render_rgb_array()[...,:3]);draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,1280,116),fill='#101820')
                draw.text((20,12),'RoundedDie | Wuji HandPose | home +/-0.5 rad | EMA=0.5 | ENVIRONMENT PREVIEW',font=font,fill='white')
                draw.text((20,48),f't={t:.2f}s | Fixed joint targets | NO TRAINING / NO POLICY',font=font,fill='#6ae0d0')
                draw.text((20,84),'Original 70 mm mesh | Newton MJWarp | 20 Hz | physics randomization enabled',font=small,fill='#c5d4df')
                draw.text((880,260),'TARGET orientation',font=font,fill='#6ae0d0')
                draw.text((880,540),'Original texture / scale 1:1',font=small,fill='#c5d4df')
                draw.text((420,710),'CURRENT / physical object',font=font,fill='white')
                return image
            for _ in range(3):initial=frame(0)
            initial.save(args.output_dir/'initial_view.png')
            writer=imageio.get_writer(str(args.output_dir/'preview.mp4'),fps=20,codec='libx264',
                macro_block_size=16,ffmpeg_params=['-crf','23','-preset','fast','-threads','4','-movflags','+faststart'])
        data=NewtonManager._solver.mjw_data
        peaks=dict(candidates=0,contacts=0,constraints=0);rows=[]
        with torch.no_grad():
            for step in range(args.steps):
                row,_,_,clean=a.step(hold)
                assert torch.isfinite(a.state).all() and torch.isfinite(row['reward']).all() and torch.isfinite(a.geometry(a.state)).all()
                assert int(np.bitwise_or.reduce(data.overflow.numpy().ravel()))==0
                peaks['candidates']=max(peaks['candidates'],int(data.ncollision.numpy().max()))
                peaks['contacts']=max(peaks['contacts'],int(data.nacon.numpy().max()))
                peaks['constraints']=max(peaks['constraints'],int(data.nefc.numpy().max()))
                done=row['done']
                if done.any():hold[done]=held_action()[done]
                entry=dict(step=step+1,resets=a.reset_count,
                    object_position=tensor(a.obj.data.root_pos_w)[0].tolist(),
                    max_linear_speed=float(tensor(a.obj.data.root_lin_vel_w).norm(dim=-1).max()),
                    max_angular_speed=float(tensor(a.obj.data.root_ang_vel_w).norm(dim=-1).max()))
                rows.append(entry)
                if viz:
                    img=frame((step+1)*env.step_dt);writer.append_data(np.asarray(img))
                    if step in (0,19,59,99,args.steps-1):img.save(args.output_dir/f'frame_{step+1:04d}.png')
                if (step+1)%50==0:print('PREVIEW_STEP',json.dumps(entry),flush=True)
        if writer:writer.close();writer=None
        # Exercise the real terminal/reset recorder without any optimizer updates.
        env.episode_length_buf[:]=env.max_episode_length-1;a.state=a.read_state()
        row,_,_,clean=a.step(hold);assert row['done'].all();a.audit_observation(clean)
        result.update(passed=True,frames=args.steps if viz else 0,seconds=args.steps*env.step_dt,
            contact_peaks=peaks,parity_max_errors=a.max_errors,resets=a.reset_count,
            forced_timeout_verified=True,trajectory=rows)
        dump_yaml(str(args.output_dir/'env.yaml'),env.cfg)
        print('DIE_PREVIEW_PASSED',json.dumps({k:v for k,v in result.items() if k not in ('geometry','trajectory','noise_config','binding','mass_kg','inertia_kg_m2')}),flush=True)
    finally:
        if writer:writer.close()
        (args.output_dir/'validation.json').write_text(json.dumps(result,indent=2))
        env.close();app.close()
if __name__=='__main__':main()
