"""Guarded two-GPU collection: 1,000-pair gate, then 20k train + 1k clean val.

Only orchestrates actor inference in this isolated checkout. No training imports,
dependency installation, external process control, or automatic failed-run reuse.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
GPUS = ['GPU-c353bd61-5a0a-74b4-de6b-91243ff9931b',
        'GPU-ac561d84-ce6e-f1ac-82ee-8c9dc3316811']
PYTHON = '/home/zhangke/viserDex_revo3/.venv-newton/bin/python'
CALIBRATION = 'config/cameras/realsense_261722074750_1280x720_extrinsics_revo3_20261101.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--preflight', type=Path, default=ROOT.parent/'evidence/yellow_preflight.json')
    p.add_argument('--parallel_envs', type=int, default=1)
    p.add_argument('--parallel_validation', type=Path)
    p.add_argument('--reuse_dataset', type=Path)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = out/'evidence'
    evidence.mkdir()
    preflight = json.loads(args.preflight.read_text())
    protected = preflight['protected_files']
    source = Path(preflight['source_root'])

    def check_source():
        for path, expected in protected.items():
            assert digest(source/path) == expected, 'Protected source changed: '+path

    check_source()
    plan, reused = [], []
    verified = dict(train=0, validation=0)
    if args.reuse_dataset:
        previous = json.loads((args.reuse_dataset/'manifest.json').read_text())
        assert previous['policy_sha256'] == preflight['identity']['checkpoint_sha256']
        assert previous['calibration_sha256'] == digest(ROOT/CALIBRATION)
        for original in previous['shards']:
            path = args.reuse_dataset/original['path']
            validation = args.reuse_dataset/original['validation']
            result = json.loads(validation.read_text())
            assert result['passed'] and result['samples'] == original['samples']
            assert digest(path/'manifest.json') == original['manifest_sha256']
            record = dict(original, path=os.path.relpath(path,out),
                          validation=os.path.relpath(validation,out), reused=True, num_envs=1)
            reused.append(record)
            verified[record['split']] += record['samples']
    if args.parallel_envs > 1:
        check = json.loads(args.parallel_validation.read_text())
        assert check['passed'] and check['num_envs'] == args.parallel_envs
        assert not check['policy_audit']['contact_overflow']
        for split, target, base_seed in [('train',20000,8200),('validation',1000,9200)]:
            remaining = target-verified[split]
            assert remaining >= 0
            for i in range(2):
                count = remaining//2 + int(i < remaining%2)
                if count:
                    plan.append(dict(id=f'{split}_parallel_{i:03d}', split=split, seed=base_seed+i,
                        augmentation_seed=30000000+(base_seed+i)*1000000,
                        samples=count, gpu_slot=i, pilot=False, num_envs=args.parallel_envs))
    else:
        assert not reused, 'Reuse is supported by the parallel plan'
        for split, count, base_seed in [('train', 40, 6100), ('validation', 2, 7100)]:
            for index in range(count):
                plan.append(dict(id=f'{split}_{index:03d}', split=split, seed=base_seed+index,
                    augmentation_seed=30000000+(base_seed+index)*10000,
                    samples=500, gpu_slot=index%2, pilot=split=='train' and index<2, num_envs=1))
    assert len({j['seed'] for j in plan}) == len(plan)
    assert not ({j['seed'] for j in plan} & {j['seed'] for j in reused})
    code = ['scripts/pose_estimator/collect_yellow_dataset.py',
            'scripts/pose_estimator/collect_yellow_die_rgbd.py',
            'scripts/pose_estimator/validate_yellow_capture.py',
            'source/viserdex/viserdex/collectors/yellow_die_rgbd.py',
            'source/viserdex/viserdex/collectors/yellow_parallel.py',
            'source/viserdex/viserdex/collectors/occlusion_audit.py',
            'source/viserdex/viserdex/collectors/point_noise.py',
            'source/viserdex/viserdex/renderer/splat_randomization.py']
    hashes = {name:digest(ROOT/name) for name in code}
    manifest = dict(version=2, status='collecting' if args.parallel_envs>1 else 'pilot', completed=False, training_ready=False,
        requested=dict(train=20000, validation=1000), verified=verified,
        control_hz=20, capture_hz=10, stride=2, burn_in_policy_steps=40,
        num_envs_per_gpu=args.parallel_envs, physical_gpus=[2,4], gpu_uuids=GPUS,
        reused_dataset=str(args.reuse_dataset) if args.reuse_dataset else None,
        parallel_validation=str(args.parallel_validation) if args.parallel_validation else None,
        plan=plan, code_sha256=hashes, protected_source_sha256=protected,
        calibration_sha256=digest(ROOT/CALIBRATION),
        policy_sha256=preflight['identity']['checkpoint_sha256'],
        split_policy='Disjoint scene/trajectory seeds; validation has clean SH and clean points',
        input_keys=dict(train=['rgb','points_camera_m','point_uv_rgb','point_valid'],
                        validation=['rgb_clean','points_camera_clean_m','point_uv_rgb','point_valid']),
        audit_only_keys=['audit_isolated_depth_m'],
        label_keys=['T_camera_object','T_base_object','keypoints_camera_m','keypoints_object_m'],
        limitations=['Synthetic data and virtual co-located depth',
                     'Visibility quality gate rejects <200 depth or <32 RGB target pixels',
                     'Coverage follows this policy; no real-camera benchmark'],
        started_unix=time.time(), orchestrator_pid=os.getpid(), running={}, shards=reused)
    lock = threading.Lock()
    stopped = threading.Event()

    def save():
        manifest['updated_unix'] = time.time()
        tmp = out/'manifest.json.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(out/'manifest.json')

    save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MAX_JOBS='2',
        VISERDEX_PYTHON=PYTHON, ISAACLAB_PATH='/home/zhangke/IsaacLab-v3.0.0',
        UV_BIN='/home/zhangke/isaaclab-ga-upgrade-20261020/bin/uv',
        TORCH_EXTENSIONS_DIR=str(ROOT/'.cache/torch_extensions'),
        PYTHONDONTWRITEBYTECODE='1')

    def run_shard(job):
        check_source()
        for name, expected in hashes.items():
            assert digest(ROOT/name) == expected, 'Collector changed during run: '+name
        shard = out/job['split']/job['id']
        gpu = GPUS[job['gpu_slot']]
        state = evidence/(job['id']+'_state.json')
        command = ['env', 'CUDA_VISIBLE_DEVICES='+gpu, 'nice', '-n', '10',
            'bash', 'scripts/run_yellow_die.sh', 'scripts/pose_estimator/collect_yellow_die_rgbd.py',
            '--num_envs',str(job['num_envs']),'--viz','none','--env_config','configs/rgbd_capture/env.yaml',
            '--policy_env_config','configs/rgbd_capture/policy_original_env.yaml',
            '--policy_checkpoint','checkpoints/policy/yellow_actor_32100.pt',
            '--calibration',CALIBRATION,'--fixture','assets/calibration_fixture/calibration_fixture_visual.usda',
            '--output_dir',str(shard),'--samples',str(job['samples']), '--stride','2',
            '--seed',str(job['seed']),'--augmentation_seed',str(job['augmentation_seed']),
            '--formal','--burn_in','40','--split',job['split'], '--render_batch_size','8','--writer_threads','4']
        if job['split'] == 'validation':
            command += ['--no_splat_augmentation','--no_point_noise']
        guard = ['python3','scripts/supermimic/run_guard.py','--gpu',gpu,'--run_dir',str(shard),
            '--state_file',str(state),'--console',str(evidence/(job['id']+'.log')),'--']+command
        with lock:
            manifest['running'][str(job['gpu_slot'])] = job['id']
            save()
        with (evidence/(job['id']+'_guard.log')).open('x') as log:
            subprocess.run(guard, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # A shard is counted only after every saved pair has been read and audited.
        validation = evidence/(job['id']+'_validation.json')
        with (evidence/(job['id']+'_validation.log')).open('x') as log:
            subprocess.run(['nice','-n','10','bash','scripts/run_yellow_die.sh',
                'scripts/pose_estimator/validate_yellow_capture.py','--dataset',str(shard),
                '--preflight',str(args.preflight),'--calibration',CALIBRATION,
                '--output',str(validation)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(validation.read_text())
        assert result['passed'] and result['samples'] == job['samples']
        check_source()
        m = json.loads((shard/'manifest.json').read_text())
        record = dict(**job, path=str(shard.relative_to(out)), validation=str(validation.relative_to(out)),
            manifest_sha256=digest(shard/'manifest.json'), accepted=m['samples'], rejected=m['rejected'],
            policy_steps=m['physics_steps'], episodes=m['episodes'],
            occlusion=result['occlusion_audit'], valid_points_min=result['valid_points_min'])
        with lock:
            manifest['shards'].append(record)
            manifest['verified'][job['split']] += m['samples']
            del manifest['running'][str(job['gpu_slot'])]
            save()
        print(json.dumps(dict(completed_shard=job['id'], verified=manifest['verified'])), flush=True)

    def worker(jobs):
        for job in jobs:
            if stopped.is_set():
                return
            try:
                run_shard(job)
            except BaseException:
                stopped.set()
                raise

    try:
        # Both pilot shards must pass before either GPU starts the main run.
        if args.parallel_envs == 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, [j for j in plan if j['pilot'] and j['gpu_slot']==i]) for i in range(2)]
                for f in futures:
                    f.result()
            assert manifest['verified']['train'] == 1000
            assert max(s['occlusion']['fraction_max'] for s in manifest['shards']) > .05
        manifest.update(status='collecting', pilot_passed=True, pilot_passed_unix=time.time())
        save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, [j for j in plan if not j['pilot'] and j['gpu_slot']==i]) for i in range(2)]
            for f in futures:
                f.result()
        assert manifest['verified'] == manifest['requested']
        for split in ['train', 'validation']:
            with (out/(split+'_index.jsonl')).open('x') as index:
                for shard in sorted(manifest['shards'], key=lambda s:s['id']):
                    if shard['split'] != split:
                        continue
                    m = json.loads((out/shard['path']/'manifest.json').read_text())
                    assert digest(out/shard['path']/'manifest.json') == shard['manifest_sha256']
                    for frame in m['frames']:
                        row = dict(uid=f"{shard['id']}/{frame['frame']:05d}",
                            pair=f"{shard['path']}/{frame['frame']:05d}/pair.npz", split=split,
                            seed=shard['seed'], sequence=frame['sequence'],
                            env_id=frame.get('env_id',0),
                            policy_step=frame['policy_step'], pair_sha256=frame['pair_sha256'],
                            occlusion_fraction=frame['occlusion_audit']['occlusion_fraction'])
                        index.write(json.dumps(row)+'\n')
        check_source()
        manifest.update(status='completed', completed=True, training_ready=True,
                        finished_unix=time.time(), index_sha256={s:digest(out/(s+'_index.jsonl')) for s in ['train','validation']})
        save()
    except BaseException as exc:
        # Let an already running sibling finish its own shard; never touch other jobs.
        manifest.update(status='failed', error=repr(exc), finished_unix=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
