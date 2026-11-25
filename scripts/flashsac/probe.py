"""Read-only frozen-checkpoint action/entropy/Q diagnostics on saved observations."""
import argparse
import json
from pathlib import Path
import torch
from flash_rl.agents.flashSAC.agent import FlashSACConfig
from viserdex.supermimic.core import Config
from viserdex.flashsac.agent import RevoFlashSAC, load_evaluation, tree_hash
from viserdex.flashsac.observations import ObservationEncoder

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261106)
    before=tree_hash(args.checkpoint)
    meta=json.loads((args.checkpoint/'metadata.json').read_text())
    cfg=Config(**meta['task_config']['config'])
    algorithm=dict(meta['algorithm'],device_type='cpu',buffer_device_type='cpu',
                   buffer_max_length=1,buffer_min_length=1)
    agent=RevoFlashSAC(FlashSACConfig(**algorithm),cfg,action_mode=meta.get('action_mode','s0'))
    encoder=ObservationEncoder(cfg,torch.ones(65))
    load_evaluation(args.checkpoint,agent,encoder,meta['environment_contract'])
    observations=torch.load(args.observations,weights_only=True,map_location='cpu')
    output=dict(checkpoint=str(args.checkpoint),observations=str(args.observations),groups={})
    limits=torch.tensor(cfg.network_joint_limits);half=(limits[:,1]-limits[:,0])*.5
    home=torch.tensor(cfg.action_home)-torch.tensor(cfg.encoder_bias_rad)
    with torch.no_grad():
        alpha=float(agent._temperature())
        norm=agent.reward_normalizer
        denom=float(torch.maximum(torch.sqrt(norm.G_rms.var+norm.epsilon),norm.G_r_max/norm.G_max))
        for name,obs in observations.items():
            mean,std=agent._actor.network.get_mean_and_std(obs[:,:810],False)
            expanded=obs.repeat(32,1)
            actions,info=agent._actor(observations=expanded,training=False)
            qs,qinfo=agent._target_critic(observations=expanded,actions=actions,training=False)
            probs=qinfo['log_prob'].exp()
            mean_action=agent.act(obs,training=False)
            previous=torch.where(obs[:,897:898].bool(),obs[:,867:888],obs[:,810:831])
            output['groups'][name]=dict(
                observations=len(obs),pre_tanh_std_mean=float(std.mean()),
                pre_tanh_std_p95=float(torch.quantile(std,.95)),
                local_mean_abs=float(mean.tanh().abs().mean()),
                deterministic_target_step_abs_deg=float(((mean_action-previous)*half).abs().mean()*180/torch.pi),
                deterministic_home_offset_abs_deg=float((mean_action*half+limits.mean(-1)-home).abs().mean()*180/torch.pi),
                local_entropy_mc=float(-info['log_prob'].mean()),target_entropy=agent._cfg.temp_target_entropy,
                alpha=alpha,entropy_bonus_mean=float(-alpha*info['log_prob'].mean()),
                reward_denominator=denom,target_q_mean=float(qs.mean()),
                target_lower_edge_probability=float(probs[...,0].mean()),
                target_upper_edge_probability=float(probs[...,-1].mean()))
    assert before==tree_hash(args.checkpoint)
    output['source_checkpoint_unchanged']=True
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
