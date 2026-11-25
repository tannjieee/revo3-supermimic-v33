"""Batched Actor play and bounded RGB-D processing/writing for many worlds."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from .yellow_die_rgbd import tensor, digest


def cpu(value):
    return value.detach().cpu().numpy()


def transforms(position, quaternion):
    result = np.tile(np.eye(4), (len(position), 1, 1))
    result[:, :3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:, :3, 3] = position
    return result


def write_pair(folder, arrays, scene, row):
    folder.mkdir()
    for key, name in [('rgb', 'rgb.png'), ('rgb_clean', 'rgb_clean.png'), ('play_raw_rgb', 'play_raw.png')]:
        Image.fromarray(arrays[key]).save(folder/name)
    Image.fromarray(scene).save(folder/'scene.png')
    np.savez_compressed(folder/'pair.npz', **arrays)
    row['pair_sha256'] = digest(folder/'pair.npz')
    (folder/'metadata.json').write_text(json.dumps(row, indent=2))
    return row


@torch.no_grad()
def collect_parallel(env, args, actor, adapter, capture, manifest, out):
    from isaaclab_newton.physics import NewtonManager
    from isaaclab.utils.io import dump_yaml
    assert actor is not None and args.formal
    n = env.num_envs
    manifest.update(formal=True, num_envs=n, split=args.split, requested_samples=args.samples,
        capture_fps=1/(env.step_dt*args.stride), stride=args.stride, burn_in_steps=args.burn_in,
        attempts=0, rejected=0, rejection_reasons={}, started_unix=time.time(),
        capture=capture.contract, samples=0, physics_steps=0, completed=False,
        parallel=dict(policy_batch=n, camera_worlds=n, gaussian_processing_batch=args.render_batch_size,
                      writer_threads=args.writer_threads, max_pending_writes=4*args.writer_threads),
        limitation='Synthetic parallel policy trajectories; virtual co-located depth; no real-sensor evaluation')
    episode = np.zeros(n, dtype=np.int64)
    episode_steps = np.zeros(n, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    # Distribute quotas over the full grid, including held-out 500/1000 captures.
    quota = np.full(n, args.samples//n, dtype=np.int64)
    remainder = args.samples % n
    if remainder:
        quota[np.linspace(0, n-1, remainder, dtype=int)] += 1
    initial_joints = cpu(tensor(env.scene['robot'].data.joint_pos)).copy()
    pending, rows, rollout = deque(), [], []
    total_steps = issued = rounds = 0
    failures = (out/'rejected.jsonl').open('w')
    transitions = (out/'policy_transitions.jsonl').open('w')
    max_policy_steps = args.burn_in + 400 + 20*args.stride*int(quota.max())

    def progress():
        manifest.update(samples=len(rows), issued_samples=issued, physics_steps=total_steps,
            simulated_environment_steps=total_steps*n, episodes=int(episode.sum()+n),
            environments_with_samples=int((counts > 0).sum()),
            updated_unix=time.time(), capture=capture.contract, capture_rounds=rounds)
        tmp = out/'manifest.json.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(out/'manifest.json')

    def commit_one():
        rows.append(pending.popleft().result())

    def advance(steps):
        nonlocal total_steps
        for _ in range(steps):
            action = actor(adapter.history)
            assert action.shape == (n,21) and torch.isfinite(action).all()
            transition, _, _, _ = adapter.step(action)
            assert torch.isfinite(transition['next_physical']).all()
            assert torch.isfinite(transition['reward']).all()
            assert int(np.bitwise_or.reduce(NewtonManager._solver.mjw_data.overflow.numpy().ravel())) == 0, 'Newton capacity overflow'
            done = cpu(transition['done']).astype(bool)
            success = cpu(transition['success']).astype(bool)
            rollout.append(dict(action=cpu(action), done=done, success=success,
                reward=cpu(transition['reward']), orientation_error_rad=cpu(adapter.captured_orientation_error),
                angular_speed_rad_s=cpu(adapter.captured_angular_speed)))
            total_steps += 1
            episode_steps[:] += 1
            episode[:] += done
            episode_steps[done] = 0
            transitions.write(json.dumps(dict(step=total_steps, environments=n, resets=int(done.sum()),
                reached=int(success.sum()), reward_mean=float(transition['reward'].mean())))+'\n')
        transitions.flush()

    progress()
    try:
        with ThreadPoolExecutor(max_workers=args.writer_threads, thread_name_prefix='rgbd_writer') as writer:
            advance(args.burn_in)
            while issued < args.samples:
                if total_steps > max_policy_steps or manifest['attempts'] >= 10*args.samples:
                    raise RuntimeError('Parallel replenishment limit reached; incomplete dataset')
                eligible = np.flatnonzero((counts < quota) & (episode_steps >= args.burn_in))
                action = actor(adapter.history)
                assert torch.isfinite(action).all()
                first_chunk = True
                for offset in range(0, len(eligible), args.render_batch_size):
                    ids = eligible[offset:offset+args.render_batch_size]
                    ids_t = torch.as_tensor(ids, device=env.device)
                    rgb, points, uv, valid, visible_counts = capture.capture(ids_t, refresh_scene=first_chunk)
                    first_chunk = False
                    f = capture.last_frame
                    # Transfer only this bounded image batch; native cameras keep their buffers on GPU.
                    data = {key:cpu(f[key]) for key in ['scene_rgb','rgb_clean','play_raw_rgb','depth_m',
                        'rgb_visible','depth_visible','points_camera_clean_m','point_depth_noise_m',
                        'K_rgb','K_depth','isolated_depth_m','camera_pos','camera_quat','object_pos','object_quat']}
                    data.update(rgb=cpu(rgb.permute(0,2,3,1).mul(255).round().byte()),
                        points_camera_m=cpu(points), point_uv_rgb=cpu(uv), point_valid=cpu(valid))
                    robot = env.scene['robot'].data
                    twc = transforms(data['camera_pos'], data['camera_quat'])
                    two = transforms(data['object_pos'], data['object_quat'])
                    twb = transforms(cpu(tensor(robot.root_pos_w)[ids_t]), cpu(tensor(robot.root_quat_w)[ids_t]))
                    tco, tbo = np.linalg.inv(twc) @ two, np.linalg.inv(twb) @ two
                    joints, actions = cpu(tensor(robot.joint_pos)[ids_t]), cpu(action[ids_t])
                    for i, eid in enumerate(ids):
                        audit = f['occlusion_audit'][i]
                        manifest['attempts'] += 1
                        if not audit['passed']:
                            manifest['rejected'] += 1
                            for reason in audit['reasons']:
                                manifest['rejection_reasons'][reason] = manifest['rejection_reasons'].get(reason,0)+1
                            failures.write(json.dumps(dict(env_id=int(eid), step=total_steps,
                                episode=int(episode[eid]), audit=audit))+'\n')
                            continue
                        sequence = int(eid + episode[eid]*n)
                        arrays = {key:data[key][i].copy() for key in ['rgb','rgb_clean','play_raw_rgb','depth_m',
                            'rgb_visible','depth_visible','points_camera_m','points_camera_clean_m',
                            'point_depth_noise_m','point_uv_rgb','point_valid','K_rgb','K_depth']}
                        arrays.update(audit_isolated_depth_m=data['isolated_depth_m'][i].copy(),
                            T_world_camera=twc[i], T_world_object=two[i], T_camera_object=tco[i], T_base_object=tbo[i],
                            keypoints_object_m=capture.reference,
                            keypoints_camera_m=(capture.reference@tco[i,:3,:3].T+tco[i,:3,3]).astype('float32'),
                            joint_pos=joints[i], policy_action_to_next=actions[i],
                            sequence_id=np.int64(sequence), env_id=np.int64(eid), episode_id=episode[eid].copy(),
                            frame_id=np.int64(issued), policy_step=np.int64(total_steps),
                            sim_time_s=np.float64(total_steps*env.step_dt),
                            episode_time_s=np.float64(episode_steps[eid]*env.step_dt))
                        row = dict(frame=issued, env_id=int(eid), sequence=sequence, episode=int(episode[eid]),
                            policy_step=total_steps, splat_augmentation=f['splat_augmentation'][i],
                            point_noise=f['point_noise'][i], occlusion_audit=audit,
                            visible_depth_pixels=int(visible_counts[i]), valid_points=int(valid[i].sum()),
                            visible_rgb_pixels=int(f['rgb_visible'][i].sum()),
                            visible_fixture_rgb_pixels=f['visible_fixture_rgb_pixels'][i],
                            visible_fixture_depth_pixels=f['visible_fixture_depth_pixels'][i])
                        pending.append(writer.submit(write_pair, out/f'{issued:05d}', arrays, data['scene_rgb'][i].copy(), row))
                        counts[eid] += 1
                        issued += 1
                        if len(pending) >= 4*args.writer_threads:
                            commit_one()
                    if issued % 128 < args.render_batch_size:
                        while pending and pending[0].done():
                            commit_one()
                        progress()
                        print(json.dumps(dict(phase='capture', policy_step=total_steps,
                            samples=len(rows), issued=issued, environments=n,
                            active_envs=len(eligible), rejected=manifest['rejected'])), flush=True)
                rounds += 1
                failures.flush()
                progress()
                if issued < args.samples:
                    advance(args.stride)
            while pending:
                commit_one()
        assert np.array_equal(counts, quota)
        np.savez_compressed(out/'policy_rollout.npz', **{key:np.stack([r[key] for r in rollout]) for key in rollout[0]},
            initial_joint_pos=initial_joints, final_joint_pos=cpu(tensor(env.scene['robot'].data.joint_pos)))
        manifest['policy_audit'] = dict(parity_max_errors=adapter.max_errors, resets=adapter.reset_count,
            successes=sum(int(r['success'].sum()) for r in rollout),
            nonzero_action_steps=sum(bool(np.any(np.abs(r['action'])>1e-6)) for r in rollout),
            steps=total_steps, simulated_environment_steps=n*total_steps, contact_overflow=False,
            per_environment_joint_motion_rad=np.max(np.abs(cpu(tensor(env.scene['robot'].data.joint_pos))-initial_joints),axis=1).tolist())
        manifest.update(completed=True, frames=sorted(rows,key=lambda r:r['frame']),
            samples_per_env=counts.tolist(), finished_unix=time.time())
        progress()
        dump_yaml(str(out/'env.yaml'), env.cfg)
    except BaseException as error:
        manifest.update(completed=False, error=repr(error))
        progress()
        raise
    finally:
        failures.close()
        transitions.close()
