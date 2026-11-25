"""Read-only checks of the short yellow policy RGB-D capture and source files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training_pids', type=int, nargs='*', default=[])
    p.add_argument('--calibration', type=Path)
    a = p.parse_args()
    m = json.loads((a.dataset / 'manifest.json').read_text())
    pre = json.loads(a.preflight.read_text())
    assert m['completed'] and m['mode'] == 'policy_rollout_rgbd'
    assert m['policy']['iteration'] == 32100
    assert m['policy']['checkpoint_sha256'] == pre['identity']['checkpoint_sha256']
    assert m['capture']['splat_sha256'] == pre['identity']['splat_sha256']
    assert m['policy']['optimizer_loaded'] is False
    if a.calibration:
        assert m['capture']['calibration_sha256'] == hashlib.sha256(a.calibration.read_bytes()).hexdigest()
        assert 'passed' in m['capture']['live_camera_calibration_audit']
    assert m['policy_audit']['contact_overflow'] is False
    stride = round(m['policy']['control_hz'] / m['capture_fps'])
    if m.get('formal'):
        assert m['samples'] == m['requested_samples']
        assert m['attempts'] == m['samples']+m['rejected']
        assert m['policy_audit']['steps'] == m['physics_steps']
        if m.get('num_envs', 1) > 1:
            assert m['physics_steps'] >= m['burn_in_steps']
            assert m['parallel']['policy_batch'] == m['num_envs']
            assert m['parallel']['camera_worlds'] == m['num_envs']
            assert sum(m['samples_per_env']) == m['samples']
            with np.load(a.dataset/'policy_rollout.npz') as rollout:
                assert rollout['action'].shape == (m['physics_steps'],m['num_envs'],21)
                assert np.isfinite(rollout['action']).all()
                assert np.all(np.max(np.abs(rollout['action']), axis=(0,2)) > 1e-6)
                assert rollout['done'].shape == (m['physics_steps'],m['num_envs'])
        else:
            assert m['physics_steps'] >= (m['samples']-1)*stride+m['burn_in_steps']
    else:
        assert m['policy_audit']['steps'] == (m['samples']-1)*stride
    fixture_validation = None
    if 'fixture' in m['capture']:
        fixture_validation = dict(contract=m['capture']['fixture'],
            visible_rgb_pixels_min=min(r['visible_fixture_rgb_pixels'] for r in m['frames']),
            visible_depth_pixels_min=min(r['visible_fixture_depth_pixels'] for r in m['frames']))
        assert fixture_validation['visible_rgb_pixels_min'] > 0
        assert fixture_validation['visible_depth_pixels_min'] > 0
    files = sorted(a.dataset.glob('[0-9][0-9][0-9][0-9][0-9]/pair.npz'))
    assert len(files) == m['samples'] and len(files) > 1
    uv_error = depth_error = transform_error = 0.0
    counts, joints, poses, changed = [], [], [], 0
    noise_values = []
    occlusions, isolated_errors = [], []
    per_env = np.zeros(m.get('num_envs',1),dtype=int)
    for f in files:
        row = json.loads((f.parent / 'metadata.json').read_text())
        assert hashlib.sha256(f.read_bytes()).hexdigest() == row['pair_sha256'], str(f)
        with np.load(f) as d:
            for key in d.files:
                values = d[key][d['depth_visible']] if key == 'depth_m' else d[key]
                assert np.isfinite(values).all(), (f.name, key)
            valid = d['point_valid']
            points = d['points_camera_m'][valid]
            assert len(points) > 0
            assert not d['rgb'][~d['rgb_visible']].any()
            assert not d['rgb_clean'][~d['rgb_visible']].any()
            assert not d['points_camera_m'][~valid].any()
            projected = points @ d['K_rgb'].T
            err = np.max(np.abs(projected[:, :2] / projected[:, 2:] - d['point_uv_rgb'][valid]))
            uv_error = max(uv_error, float(err))
            clean_points = d['points_camera_clean_m'][valid] if 'points_camera_clean_m' in d else points
            delta = points[:,2] - clean_points[:,2]
            if m['capture'].get('point_noise'):
                assert np.max(np.abs(delta)) <= m['capture']['point_noise']['depth_clip_m'] + 1e-7
                assert np.allclose(points / points[:,2:], clean_points / clean_points[:,2:], atol=1e-6, rtol=0)
                assert np.array_equal(delta, d['point_depth_noise_m'][valid])
                noise_values.extend(delta.tolist())
            else:
                assert np.array_equal(points, clean_points)
            pd = clean_points @ d['K_depth'].T
            coords = pd[:, :2] / pd[:, 2:]
            xy = (np.floor(coords) if m['capture'].get('pixel_coordinates') else np.rint(coords)).astype(int)
            assert d['depth_visible'][xy[:, 1], xy[:, 0]].all()
            depth_error = max(depth_error, float(np.abs(d['depth_m'][xy[:, 1], xy[:, 0]] - clean_points[:, 2]).max()))
            transform_error = max(transform_error, float(np.abs(d['T_world_camera'] @ d['T_camera_object'] - d['T_world_object']).max()))
            expected_keypoints = d['keypoints_object_m'] @ d['T_camera_object'][:3,:3].T + d['T_camera_object'][:3,3]
            assert np.allclose(expected_keypoints, d['keypoints_camera_m'], atol=1e-6, rtol=0)
            if m.get('formal'):
                eid = int(d['env_id']) if 'env_id' in d else 0
                assert 0 <= eid < len(per_env)
                per_env[eid] += 1
                if m.get('num_envs', 1) > 1:
                    assert int(d['sequence_id']) == eid+int(d['episode_id'])*m['num_envs']
                    assert int(d['policy_step']) == row['policy_step'] <= m['physics_steps']
                    assert eid == row['env_id']
                assert float(d['episode_time_s']) >= m['burn_in_steps']/m['policy']['control_hz']-1e-6
                audit = row['occlusion_audit']
                assert audit['passed'] and not audit['reasons']
                isolated = d['audit_isolated_depth_m']
                mask = d['depth_visible']
                assert (isolated[mask] > 0).all()
                error = np.abs(isolated[mask] - d['depth_m'][mask]).max()
                assert error <= m['capture']['occlusion_audit']['depth_tolerance_m']
                assert mask.sum() >= 200 and d['rgb_visible'].sum() >= 32
                silhouette = isolated > 0
                fraction = (silhouette & ~mask).sum()/max(1, silhouette.sum())
                assert abs(fraction-audit['occlusion_fraction']) < 1e-7
                isolated_errors.append(float(error))
                occlusions.append(float(fraction))
            counts.append(len(points))
            joints.append(d['joint_pos'].copy())
            poses.append(d['T_world_object'].copy())
            changed += int(np.any(d['rgb'] != d['rgb_clean']))
    assert uv_error < 1e-4 and depth_error < 1e-6 and transform_error < 1e-6
    if m.get('num_envs',1) > 1:
        assert per_env.tolist() == m['samples_per_env']
    joint_motion = float(np.ptp(np.array(joints), axis=0).max())
    object_motion = float(np.ptp(np.array(poses), axis=0).max())
    assert joint_motion > .01 and object_motion > .01
    if m['capture']['splat_augmentation']:
        assert changed > 0
    else:
        assert changed == 0
    noise_audit = None
    if noise_values:
        noise_values = np.asarray(noise_values)
        assert .8*m['capture']['point_noise']['depth_std_m'] < noise_values.std() < 1.2*m['capture']['point_noise']['depth_std_m']
        noise_audit = dict(std_m=float(noise_values.std()), max_abs_m=float(np.abs(noise_values).max()),
                           mean_m=float(noise_values.mean()), config=m['capture']['point_noise'])
    protected = {}
    for name, expected in pre['protected_files'].items():
        actual = hashlib.sha256((Path(pre['source_root']) / name).read_bytes()).hexdigest()
        assert actual == expected, name
        protected[name] = actual
    pids = a.training_pids
    process_rows = (subprocess.check_output(['ps', '-p', ','.join(map(str,pids)), '-o', 'pid=,etimes=,args='], text=True) if pids else '')
    alive = [int(line.split()[0]) for line in process_rows.splitlines()]
    assert sorted(alive) == sorted(pids)
    result = dict(passed=True, samples=len(files), num_envs=m.get('num_envs',1),
                  environments_with_samples=int((per_env>0).sum()),
                  samples_per_env=per_env.tolist(), policy=m['policy'], policy_audit=m['policy_audit'],
                  max_rgb_projection_error_px=uv_error, max_depth_error_m=depth_error,
                  max_pose_composition_error=transform_error, valid_points_min=min(counts), valid_points_max=max(counts),
                  joint_motion_max_rad=joint_motion, object_transform_motion_max=object_motion,
                  sh_changed_frames=changed, protected_source_hashes_unchanged=protected,
                  original_training_pids_alive=alive, original_training_processes=process_rows,
                  calibration_sha256=m['capture']['calibration_sha256'],
                  camera_calibration_audit=m['capture'].get('live_camera_calibration_audit'),
                  fixture_validation=fixture_validation,
                  point_noise_audit=noise_audit,
                  occlusion_audit=dict(max_depth_error_m=max(isolated_errors),
                      fraction_min=min(occlusions), fraction_max=max(occlusions),
                      fraction_quantiles=np.quantile(occlusions, [0,.25,.5,.75,1]).tolist(),
                      rejected=m['rejected'], rejection_reasons=m['rejection_reasons']) if occlusions else None,
                  limitation="Synthetic policy rollout and virtual co-located depth; no real-sensor evaluation")
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['policy','protected_source_hashes_unchanged','original_training_processes']}, indent=2))


if __name__ == '__main__':
    main()
