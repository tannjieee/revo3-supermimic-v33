"""Read back all configured joint limits and run a bounded holding diagnostic."""
import argparse
import json
from pathlib import Path
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.manual_convex import TASK_ID
from viserdex.supermimic.revised_contract import configure_environment, CENTER_REVISION
from viserdex.supermimic.simulation import TransitionRecorderCfg

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs',type=int,default=1024)
parser.add_argument('--output',type=Path,required=True)
add_runtime_args(parser,TASK_ID)
args=parse_runtime_args(parser)
def configure(cfg):
    configure_environment(cfg,CENTER_REVISION)
    configure_clean_state_observations(cfg)
    cfg.recorders=TransitionRecorderCfg()
    cfg.scene.clone_in_fabric=False
    cfg.scene.robot.spawn.activate_contact_sensors=False
    cfg.scene.object.spawn.activate_contact_sensors=False
    cfg.commands.object_pose.debug_vis=False
    cfg.seed=42
app=SimulationRuntime(args,configure=configure)

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from viserdex.supermimic.simulation import Adapter,tensor
from viserdex.supermimic.core import hand_center_config
from viserdex.supermimic.context import policy_order_parameter
from viserdex.supermimic.output_limits import load_output_limits
from viserdex.distillation.config import PerceptionNoiseCfg

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    env=gym.make(TASK_ID,cfg=app.cfg).unwrapped
    try:
        env.reset()
        ad=Adapter(env,PerceptionNoiseCfg(),policy_cfg=hand_center_config())
        profile=load_output_limits()
        torque=policy_order_parameter(ad.robot,'fingers','clamping','max_effort',ad.joint_names)
        velocity=tensor(ad.robot.data.joint_vel_limits)[:,ad.ids]
        torque_solver=tensor(ad.robot.data.joint_effort_limits)[:,ad.ids]
        for value,expected in ((torque,1.28),(torque_solver,1.28),(velocity,profile['velocity_limit_rad_s'])):
            torch.testing.assert_close(value,torch.full_like(value,expected),atol=2e-6,rtol=0)
        result=dict(status='limit_readback_passed',profile=profile,worlds=env.num_envs,
            joint_names=ad.joint_names,readback_count=torque.numel(),
            torque_limit_range_Nm=[float(torque.min()),float(torque.max())],
            declared_velocity_limit_range_rad_s=[float(velocity.min()),float(velocity.max())],
            velocity_hard_clamp_verified=False,identified_actuator_audit=ad.identified_hand_audit,
            hand_geometry_audit=ad.manual_asset_audit,object_geometry_audit=ad.convex_audit,
            scope='configuration and bounded holding verification; not a speed-hard-clamp or stability certificate')
        q=ad.state[:,:21]-ad.state.new_tensor(ad.hold_cfg.encoder_bias_rad)
        lo,hi=ad.limits[...,0],ad.limits[...,1]
        action=(2*(q.maximum(lo).minimum(hi)-lo)/(hi-lo)-1)
        peaks=[]
        with torch.no_grad():
            for _ in range(4):
                ad.step(action)
                applied=tensor(ad.robot.data.applied_torque)[:,ad.ids]
                assert float(applied.abs().max())<=1.28+1e-6
                peaks.append(float(ad.captured_physical[:,21:42].abs().max()))
        result.update(status='passed',control_steps=4,
            observed_control_boundary_speed_peaks_rad_s=peaks,solver_diagnostic=ad._solver_diagnostic())
        assert result['solver_diagnostic']['overflow_mask']==0
        args.output.parent.mkdir(parents=True,exist_ok=True)
        config_path=args.output.with_suffix('.env.yaml')
        dump_yaml(str(config_path),env.cfg)
        result['validated_environment_yaml']=str(config_path)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('hand_geometry_audit','object_geometry_audit','identified_actuator_audit')},indent=2))
    finally:
        env.close();app.close()
if __name__=='__main__':main()
