"""Bounded 1-step model-free train/play backend for the fixed 47g Newton task."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import tarfile

REPO = Path(__file__).resolve().parents[4]
DEFAULT = REPO/'configs/flashsac_die47'


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def lifecycle(output_dir, stage):
    with (output_dir/'lifecycle.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.time()))+'\n')
        stream.flush()


def run(argv=None):
    from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
    from viserdex.distillation.environment import SavedEnvironmentBinding
    from viserdex.distillation.teacher import configure_clean_state_observations
    from viserdex.supermimic.manual_convex import TASK_ID
    from viserdex.supermimic.simulation import TransitionRecorderCfg
    from viserdex.supermimic.revised_contract import configure_environment
    from viserdex.flashsac.agent import UPSTREAM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--env_config', type=Path, default=DEFAULT/'env.yaml')
    parser.add_argument('--task_config', type=Path, default=DEFAULT/'task.json')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--warmup_steps', type=int, default=98)
    parser.add_argument('--capacity', type=int, default=100_000)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--updates_per_step', type=int, default=2)
    parser.add_argument('--wall_seconds', type=float, default=1200.)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--exploration_eval', action='store_true',
                        help='Frozen-checkpoint stochastic evaluation, no optimizer updates')
    parser.add_argument('--noise_repeat_max', type=int, choices=(1,16),
                        help='Explicit single-factor exploration duration override')
    parser.add_argument('--audit_boundaries', action='store_true')
    parser.add_argument('--save_replay', action='store_true')
    add_runtime_args(parser, TASK_ID)
    sys.argv = [sys.argv[0], *(argv or [])]
    args = parse_runtime_args(parser)
    if args.task != TASK_ID or args.device != 'cuda:0':
        parser.error('Only the audited task and cuda:0 are supported')
    if args.evaluate != bool(args.checkpoint) or args.record and (not args.evaluate or args.num_envs != 1):
        parser.error('play requires checkpoint; recording requires play and one environment')
    if args.exploration_eval and (not args.evaluate or args.record):
        parser.error('exploration_eval requires non-recording play')
    if args.evaluate and args.noise_repeat_max is not None and not args.exploration_eval:
        parser.error('Noise override requires exploration_eval when evaluating')
    if args.record:
        # Launcher CLI selection overrides config visualizers, including --viz none.
        args.visualizer = ['newton_gl']
    if min(args.num_envs, args.steps, args.warmup_steps, args.batch, args.updates_per_step) < 1:
        parser.error('Counts must be positive')
    if not args.evaluate and args.capacity < args.num_envs*args.warmup_steps:
        parser.error('Replay capacity is below warmup transitions')
    if args.audit_boundaries and (args.num_envs > 64 or args.evaluate):
        parser.error('Boundary interventions are only for small interface audits')
    vendor = Path(os.environ['FLASH_SAC_VENDOR'])
    sha = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != UPSTREAM or subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain']).strip():
        raise RuntimeError('Pinned FlashSAC source changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = sorted(set([*Path(__file__).parent.glob('*.py'),
        *(REPO/'source/viserdex/viserdex/supermimic').rglob('*.py'),
        *(REPO/'scripts/flashsac').glob('*'), *DEFAULT.glob('*')]))
    manifest = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources if p.is_file()}
    write_json(args.output_dir/'source_manifest.json',dict(upstream=UPSTREAM,files=manifest))
    with tarfile.open(args.output_dir/'source_snapshot.tar.gz','w:gz') as archive:
        for relative in manifest: archive.add(REPO/relative,arcname=relative)
    resource = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    write_json(args.output_dir/'resource.json', dict(free_mib=resource.strip(),
        existing_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'], text=True), action='No existing process stopped'))
    if float(resource.splitlines()[0]) < (4000 if args.evaluate else 8000):
        raise RuntimeError('Insufficient free GPU memory; existing processes are preserved')
    binding = SavedEnvironmentBinding(args.env_config)

    def configure(cfg):
        configure_environment(cfg, 'hardware_center25_v3')
        binding.configure(cfg, args)
        configure_clean_state_observations(cfg)
        cfg.recorders = TransitionRecorderCfg()
        cfg.scene.clone_in_fabric = False
        cfg.scene.robot.spawn.activate_contact_sensors = False
        cfg.scene.object.spawn.activate_contact_sensors = False
        cfg.commands.object_pose.debug_vis = False
        cfg.seed = args.seed
        if args.record:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
            view = NewtonGLVisualizerCfg(headless=True, window_width=960, window_height=720,
                eye=(-.22, -.24, .78), lookat=(.02, 0., .565), streaming_view=False, enable_picking=False)
            cfg.sim.visualizer_cfgs = [view]
            cfg.sim.default_visualizer_cfg = view
        binding.verify(cfg)
        cfg.sim.log_dir = str(args.output_dir/'simulator')

    app = SimulationRuntime(args, configure=configure)
    try:
        return execute(args, app, binding)
    finally:
        lifecycle(args.output_dir, 'app_close_begin')
        app.close()
        lifecycle(args.output_dir, 'app_close_end')


def execute(args, app, binding):
    import copy
    import gymnasium as gym
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from isaaclab.utils.io import dump_yaml
    from viserdex.distillation.config import PerceptionNoiseCfg
    from viserdex.supermimic.core import Config
    from viserdex.supermimic.simulation import Adapter
    from viserdex.supermimic.manual_convex import TASK_ID
    from viserdex.supermimic.warmup_excitation import HomeExcitation
    from viserdex.flashsac.agent import configuration, RevoFlashSAC, save_checkpoint, load_evaluation, tree_hash
    from viserdex.flashsac.observations import ObservationEncoder
    from viserdex.flashsac.rewards import real_transition
    from viserdex.flashsac.exploration import audit_action, ActionRejected
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = json.loads(args.task_config.read_text())
    cfg, noise = Config(**data['config']), PerceptionNoiseCfg(**data['noise_config'])
    if cfg.control_dt != .04 or cfg.history != 10 or not noise.enabled or not noise.joint_enabled:
        raise ValueError('Task timing/history/noise contract mismatch')
    if args.evaluate:
        saved = json.loads((args.checkpoint/'metadata.json').read_text())
        if saved['task_config'] != data:
            raise ValueError('Checkpoint task/noise config mismatch')
    env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
    writer = SummaryWriter(str(args.output_dir/'tensorboard'))
    video = None
    metric_file = (args.output_dir/'metrics.jsonl').open('w')
    status, steps, transitions = 'running', 0, 0
    completed = successes = drops = timeouts = accepted = 0
    angle_sum = reward_sum = 0.
    start = time.monotonic()
    try:
        clean, _ = env.reset()
        adapter = Adapter(env, noise, cfg.history, policy_cfg=cfg)
        adapter.capture_terminal_history = True
        adapter.audit_observation(clean)
        encoder = ObservationEncoder(cfg, adapter.context.abs().mean(0).clamp_min(1e-12)).to(env.device)
        agent_cfg = configuration(str(env.device), args.seed, (args.steps-args.warmup_steps)*args.updates_per_step,
            capacity=(1 if args.evaluate else args.capacity), warmup=(1 if args.evaluate else args.num_envs*args.warmup_steps), batch=args.batch)
        if args.evaluate:
            from flash_rl.agents.flashSAC.agent import FlashSACConfig
            values = dict(saved['algorithm'], buffer_max_length=1, buffer_min_length=1)
            agent_cfg = FlashSACConfig(**values)
        if args.noise_repeat_max is not None:
            agent_cfg.actor_noise_zeta_max = args.noise_repeat_max
        agent = RevoFlashSAC(agent_cfg, cfg, action_mode=saved.get("action_mode", "s0") if args.evaluate else "s1")
        metadata = dict(environment_contract=binding.contract_sha, environment=binding.verify(app.cfg),
            task_config=data, observation_dim=963, actor_dim=810, action_contract=(f'absolute_S1_local_tanh_{agent.target_rate_deg_s:g}deg_s_entropy_local' if agent.action_mode == 's1' else 'absolute_S0_presend_reject_60deg_s'),
            seed=args.seed, geometry_digest=adapter.geometry.digest, context_schema=adapter.context_schema,
            upstream_source=os.environ['FLASH_SAC_VENDOR'], task_id=TASK_ID,
            scope='interface_audit' if args.audit_boundaries else 'evaluation' if args.evaluate else 'training',
            argv=sys.orig_argv, arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        dump_yaml(str(args.output_dir/'env.yaml'), app.cfg)
        write_json(args.output_dir/'config.json', dict(metadata, algorithm=asdict(agent._cfg)))
        write_json(args.output_dir/'geometry.json', adapter.manual_asset_audit)
        if args.evaluate:
            source_hashes = tree_hash(args.checkpoint)
            load_evaluation(args.checkpoint, agent, encoder, binding.contract_sha)
            frozen = {name:{k:v.clone() for k,v in obj.network.state_dict().items()}
                      for name,obj in [('actor',agent._actor), ('critic',agent._critic), ('target',agent._target_critic),
                                       ('temperature',agent._temperature)]}
            frozen_normalizer = copy.deepcopy(agent.reward_normalizer)
        else:
            save_checkpoint(args.output_dir/'checkpoints/initial', agent, encoder, dict(metadata, vector_steps=0))
        warm_cfg = copy.copy(cfg); warm_cfg.wm_only_iters = args.warmup_steps
        warm = HomeExcitation(adapter.limits, warm_cfg)
        active = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
        episode_success = torch.zeros_like(active)
        violations = torch.zeros(5, device=env.device, dtype=torch.long)
        boundary_results = []
        if args.record:
            import imageio.v2 as imageio
            video = imageio.get_writer(str(args.output_dir/'rollout.partial.mp4'), fps=25, codec='libx264',
                ffmpeg_params=['-crf','23','-preset','fast','-threads','2'])
            viz = next(v for v in env.sim.visualizers if v.cfg.visualizer_type == 'newton_gl')
            env.sim.render()
            imageio.imwrite(args.output_dir/'initial_view.png', viz.render_rgb_array()[..., :3])
        last_update = {}
        for index in range(args.steps):
            if time.monotonic()-start > args.wall_seconds:
                status = 'wall_budget_exhausted'; break
            with torch.no_grad():
                obs = encoder(adapter.history, adapter.state, adapter.context)
                if args.evaluate and index == 0:
                    first_probe = obs.clone()
                action = (warm.action(index) if not args.evaluate and index < args.warmup_steps
                          else agent.act(obs, training=not args.evaluate or args.exploration_eval))
                action_report = audit_action(action, adapter.state, adapter.limits, cfg)
                if args.audit_boundaries and index == 2:
                    env.episode_length_buf[0] = env.max_episode_length-1
                if args.audit_boundaries and index == 4:
                    pose = adapter.obj.data.root_pose_w.torch[:1].clone()
                    pose[:, 2] -= .5
                    adapter.obj.write_root_pose_to_sim(pose, env_ids=torch.tensor([0],device=env.device))
                record, _, _, _ = adapter.step(action)
                transition, raw, risks, ratios = real_transition(record, adapter.context, encoder, adapter.limits, cfg)
                if not torch.equal(record['next_history'][~record['done']], adapter.history[~record['done']]):
                    raise RuntimeError('Nonterminal history differs between replay and behavior')
                if record['done'].any():
                    done = record['done']
                    reset_vel = adapter.state[done][:, list(range(21,42))+list(range(49,55))]
                    if reset_vel.abs().max() > 1e-6:
                        raise RuntimeError('Reset did not clear velocities')
                if args.audit_boundaries and index in (2,4):
                    expected = 'truncated' if index == 2 else 'terminated'
                    if not bool(record[expected][0]):
                        raise RuntimeError(f'Forced {expected} not observed')
                    boundary_results.append(dict(step=index+1, kind=expected,
                        terminal_history_differs=not torch.equal(record['next_history'][0],adapter.history[0]),
                        old_goal_preserved=True))
                    write_json(args.output_dir/'boundary_audits.json',boundary_results)
                if args.exploration_eval:
                    agent.reset_noise(record['done'])
                mask = active if args.evaluate and not args.record else torch.ones_like(active)
                episode_success |= record['success'] & mask
                finished = record['done'] & mask
                completed += int(finished.sum()); successes += int((episode_success & finished).sum())
                drops += int((record['terminated'] & mask).sum()); timeouts += int((record['truncated'] & mask).sum())
                accepted += int(mask.sum()); violations += ((ratios > 1) & mask[:,None]).sum(0)
                angle_sum += float(adapter.captured_orientation_error[mask].sum())
                reward_sum += float(record['reward'][mask].sum())
                episode_success &= ~finished
                if args.evaluate:
                    active &= ~finished
                else:
                    agent.process_transition(transition)
                    warm.reset(record['done']); agent.reset_noise(record['done'])
            steps=index+1; transitions+=args.num_envs
            if not args.evaluate and steps > args.warmup_steps:
                for _ in range(args.updates_per_step):
                    last_update = agent.update()
            if video is not None:
                env.sim.render(); video.append_data(viz.render_rgb_array()[..., :3])
            if steps % 32 == 0 or steps == 1 or steps == args.steps:
                elapsed = time.monotonic()-start
                row=dict(vector_steps=steps, transitions=transitions, q_updates=agent._update_step,
                    actor_updates=(agent._update_step+1)//2, successes=successes, drops=drops, timeouts=timeouts,
                    rotation_error_rad=angle_sum/max(accepted,1), reward=reward_sum/max(accepted,1),
                    violation_fractions=(violations/max(accepted,1)).tolist(), command=action_report,
                    elapsed_s=elapsed, transitions_s=transitions/elapsed,
                    peak_torch_memory_bytes=torch.cuda.max_memory_allocated(), **last_update)
                metric_file.write(json.dumps(row)+'\n'); metric_file.flush(); print(json.dumps(row),flush=True)
                for k,v in row.items():
                    if isinstance(v,(int,float)): writer.add_scalar(k,v,transitions)
            if not args.evaluate and steps % 500 == 0:
                checkpoint=args.output_dir/f'checkpoints/step_{steps:06d}'
                save_checkpoint(checkpoint,agent,encoder,dict(metadata,vector_steps=steps))
                record_child(checkpoint, args, steps)
            if args.evaluate and not args.record and not active.any():
                break
        if status == 'running': status = 'complete'
        if args.evaluate and not args.record and active.any(): status='incomplete_evaluation'
        if args.evaluate:
            torch.save(dict(first_observation=first_probe.cpu(),last_observation=obs.cpu()),
                       args.output_dir/'policy_probe.pt')
            for name,obj in [('actor',agent._actor),('critic',agent._critic),('target',agent._target_critic),('temperature',agent._temperature)]:
                if any(not torch.equal(v,frozen[name][k]) for k,v in obj.network.state_dict().items()):
                    raise RuntimeError('Evaluation mutated network/BN')
            for key in ('G_r','G_r_max'):
                if not torch.equal(getattr(agent.reward_normalizer,key),getattr(frozen_normalizer,key)):
                    raise RuntimeError('Evaluation mutated reward normalization')
            for key in ('mean','var','count'):
                if not torch.equal(getattr(agent.reward_normalizer.G_rms,key),getattr(frozen_normalizer.G_rms,key)):
                    raise RuntimeError('Evaluation mutated return statistics')
            if source_hashes != tree_hash(args.checkpoint): raise RuntimeError('Evaluation modified checkpoint')
        else:
            save_checkpoint(args.output_dir/'checkpoints/final',agent,encoder,dict(metadata,vector_steps=steps),replay=args.save_replay)
        if video is not None:
            video.close(); video=None
            (args.output_dir/'rollout.partial.mp4').rename(args.output_dir/'rollout.mp4')
        write_json(args.output_dir/'result.json', dict(status=status,vector_steps=steps,transitions=transitions,
            q_updates=agent._update_step,actor_updates=(agent._update_step+1)//2,
            completed_episodes=completed, successes=successes,drops=drops,timeouts=timeouts,
            violation_fractions=(violations/max(accepted,1)).tolist(), rotation_error_rad=angle_sum/max(accepted,1),
            audit_errors=adapter.max_errors,contact_peaks=adapter.contact_peaks,boundary_audits=boundary_results,
            source_checkpoint_unchanged=True if args.evaluate else None, frames=steps if args.record else 0,
            elapsed_s=time.monotonic()-start))
        return 0 if status=='complete' else 2
    except Exception as exc:
        status = 'action_rejected_before_step' if isinstance(exc, ActionRejected) else 'failed'
        write_json(args.output_dir/'result.json', dict(status=status,vector_steps=steps,transitions=transitions,
            q_updates=agent._update_step if 'agent' in locals() else 0,error=str(exc),
            last_update=last_update if 'last_update' in locals() else {},
            action_rejection=exc.report if isinstance(exc,ActionRejected) else None))
        (args.output_dir/'failure.txt').write_text(traceback.format_exc())
        if 'adapter' in locals():
            physics = (adapter.physics_failure_snapshot() if hasattr(adapter,'physics_failure') else
                       dict(phase='command_not_sent',trace=adapter.physics_health.snapshot(list(range(min(args.num_envs,8))))))
            torch.save(dict(state=adapter.state, action=action if 'action' in locals() else None,
                history=adapter.history, physics=physics),args.output_dir/'failure_state.pt')
        if 'agent' in locals() and not args.evaluate:
            save_checkpoint(args.output_dir/'checkpoints/interrupted',agent,encoder,dict(metadata,vector_steps=steps))
        raise
    finally:
        if video is not None: video.close()
        lifecycle(args.output_dir, 'writer_close_begin')
        metric_file.close(); writer.close()
        lifecycle(args.output_dir, 'env_close_begin')
        env.close()
        lifecycle(args.output_dir, 'env_close_end')


def record_child(checkpoint,args,steps):
    from viserdex.flashsac.agent import tree_hash
    before=tree_hash(checkpoint)
    dest=args.output_dir/f'videos/step_{steps:06d}'; dest.parent.mkdir(exist_ok=True)
    command=[sys.executable,str(REPO/'scripts/flashsac/cli.py'),'play','--rl_library','flashsac',
        '--checkpoint',str(checkpoint),'--output_dir',str(dest),'--env_config',str(args.env_config),
        '--task_config',str(args.task_config),'--num_envs','1','--steps','500','--seed','166',
        '--record','--device','cuda:0','--viz','newton_gl','physics=newton_mjwarp']
    with (dest.parent/f'step_{steps:06d}.log').open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    if result.returncode or before!=tree_hash(checkpoint):
        raise RuntimeError(f'Periodic recording failed: {dest}')
