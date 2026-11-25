"""Current Newton hold/cold-reset/12-kick evidence. Failed proposals stay outside the bank."""
import argparse
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.supermimic.manual_convex import TASK_ID
from viserdex.distillation.teacher import configure_clean_state_observations

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidates', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--jobs_manifest', type=Path, help='Consume append-only jobs in one official launcher process')
p.add_argument('--wait_timeout', type=float, default=1800.)
p.add_argument('--num_envs', type=int, default=4)
p.add_argument('--contexts', type=int, default=2)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--max_candidates', type=int, default=0)
p.add_argument('--merge_bank', type=Path)
p.add_argument('--preserve_reset_state', action='store_true', help='Recertify supplied states without a preparatory settle; trials still use cold reset')
p.add_argument('--env_config', type=Path, help='Bind the exact intended training environment')
p.add_argument('--solver_iterations', type=int)
p.add_argument('--solver_substeps', type=int, help='Explicit numerical experiment retaining identified actuator timing')
add_runtime_args(p, TASK_ID)
a = parse_runtime_args(p)
if a.num_envs < 1 or a.contexts < 2 or a.num_envs % a.contexts or a.max_candidates < 0:
    p.error('num_envs must be a multiple of contexts, with at least two contexts')
if bool(a.candidates) == bool(a.jobs_manifest) or (a.candidates and not a.output) or a.wait_timeout <= 0:
    p.error('Provide candidates/output or jobs_manifest, and a positive wait_timeout')
if a.solver_iterations is not None or a.solver_substeps is not None:
    if not a.env_config:
        p.error('Numerical experiments require an exact source --env_config')
    from viserdex.supermimic.solver_budget import SolverBudgetBinding
    binding = SolverBudgetBinding(a.env_config, a.solver_iterations, a.solver_substeps)
else:
    from viserdex.distillation.environment import SavedEnvironmentBinding
    binding = SavedEnvironmentBinding(a.env_config) if a.env_config else None

def configure(cfg):
    if binding is not None:
        binding.configure(cfg, a)
    configure_clean_state_observations(cfg)
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = a.seed
    if binding is not None:
        binding.verify(cfg)

app = SimulationRuntime(a, configure=configure)
import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import manual_convex_config, rotation_log, quat_mul, quat_conjugate
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.grasp_runtime import write_grasp_states, synchronize_reset
from viserdex.supermimic.grasp_bank import (SCHEMA, CERTIFICATE, THRESHOLDS, live_contract, digest,
    state_digest, trial_protocol, validate_record, deduplicate)
from viserdex.supermimic.grasp_contacts import accumulate_object_contacts


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def completed_job():
    main()
    # Written last, after a successful scene close. Partial bank/report files
    # alone are never proof that every candidate completed certification.
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(a.output.with_suffix('.done.json'), dict(schema='revo3_certification_receipt_v1',
        candidates_sha256=sha(a.candidates), bank_sha256=sha(a.output),
        report_sha256=sha(a.output.with_suffix('.report.json')),
        runner_sha256=sha(Path(__file__)), seed=a.seed, contexts=a.contexts,
        num_envs=a.num_envs))


def main():
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = a.output.with_suffix('.report.json')
    data = json.loads(a.candidates.read_text())
    if data.get('schema') != 'revo3_grasp_candidates_v1':
        raise ValueError('Expected current-USD candidates')
    records = [r for r in data['records'] if r.get('geometry_passed')]
    if a.max_candidates:
        records = records[:a.max_candidates]
    # Each job gets a fresh scene and independent startup randomization. Only
    # Python imports / the official launcher process are shared across jobs.
    job_cfg = deepcopy(app.cfg)
    job_cfg.seed = a.seed
    env = gym.make(TASK_ID, cfg=job_cfg).unwrapped
    env._supermimic_solver_override_proof = getattr(binding, 'override_proof', None)
    start = time.monotonic()
    try:
        env.reset()
        adapter = Adapter(env, PerceptionNoiseCfg(), policy_cfg=manual_convex_config())
        contract = live_contract(adapter)
        # Authored float64 and runtime float32 limits describe the same asset.
        source_assets = dict(data['assets'])
        if not np.allclose(source_assets['joint_limits_rad'], contract['assets']['joint_limits_rad'], atol=2e-7, rtol=0):
            raise ValueError('Candidate joint limits differ from the training range; regenerate candidates')
        source_assets['joint_limits_rad'] = contract['assets']['joint_limits_rad']
        if source_assets != contract['assets']:
            report_path.write_text(json.dumps(dict(candidate_assets=source_assets, live_contract=contract), indent=2))
            raise ValueError('Candidate asset contract differs from live simulation: ' + str([k for k in source_assets if source_assets[k] != contract['assets'].get(k)]))
        limits = adapter.limits[0].cpu().numpy()
        from viserdex.supermimic.grasp_candidates import ConvexQueries
        from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
        query = ConvexQueries(env.cfg.scene.robot.spawn.usd_path, dict(adapter.geometry_spec, joint_names=REVO3_JOINT_NAMES))
        dump_yaml(str(a.output.with_suffix('.env.yaml')), job_cfg)
        report = dict(contract=contract, candidate_file_sha256=__import__('hashlib').sha256(a.candidates.read_bytes()).hexdigest(),
                      candidates=len(data['records']), geometry_passed=len(records), results=[], complete=False,
                      audit=adapter.manual_asset_audit, thresholds=THRESHOLDS,
                      metric_semantics='maximum over every recorded 20Hz step and all contexts; drift from exact reset pose',
                      scope='finite sampled randomized contexts; not a universal stability certificate')
        accepted = []
        if a.merge_bank:
            previous = json.loads(a.merge_bank.read_text())
            if previous.get('schema') != SCHEMA or previous.get('contract') != contract:
                raise ValueError('Merged bank contract differs; recertification required')
            for r in previous['records']:
                validate_record(r, limits, contract=contract)
            accepted = previous['records']
            report['merged_bank_sha256'] = __import__('hashlib').sha256(a.merge_bank.read_bytes()).hexdigest()
        from isaaclab_newton.physics import NewtonManager
        mj = NewtonManager._solver.mj_model
        contact = NewtonManager._solver.mjw_data
        object_geoms = [i for i in range(mj.ngeom) if 'object' in mj.geom(i).name.lower() and (mj.geom_contype[i] or mj.geom_conaffinity[i])]
        import mujoco
        mujoco.mj_saveModel(mj, str(a.output.with_suffix('.solver.mjb')))
        # Read the live MJWarp parameters, not only the authored scene setting.
        mw = NewtonManager._solver.mjw_model
        gravity = mw.opt.gravity.numpy()
        gravity_disabled = bool(int(mw.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
        gravcomp = mw.body_gravcomp.numpy()
        object_bodies = sorted({int(mj.geom_bodyid[g]) for g in object_geoms})
        report['gravity_audit'] = dict(world_gravity_m_s2=gravity.tolist(),
            configured_gravity_m_s2=contract['gravity'], gravity_disabled=gravity_disabled,
            object_bodies=[mj.body(i).name for i in object_bodies],
            object_gravity_compensation=gravcomp[..., object_bodies].tolist(),
            object_mass_before_first_step_kg=mw.body_mass.numpy()[..., object_bodies].tolist())
        if (gravity_disabled or not np.allclose(gravity, contract['gravity'], atol=1e-6, rtol=0)
                or not np.isfinite(gravity).all() or np.any(gravcomp[..., object_bodies] != 0)):
            raise ValueError('Live solver gravity differs from the uncompensated environment contract')
        report['solver_mesh_audit'] = query.audit_solver_meshes(mj)
        report['object_contact_geoms'] = [mj.geom(i).name for i in object_geoms]
        if not object_geoms:
            raise ValueError('Cannot identify live object contact geoms')
        print(json.dumps(dict(object_contact_geoms=report['object_contact_geoms'])), flush=True)

        def audit_first_physics_step():
            # Startup model changes reach MJWarp on the first actual step.
            # This must also execute for exact-state recertification, whose
            # preserve_reset_state path deliberately skips preparatory settling.
            if 'post_first_physics_step_audit' in report:
                return
            mass_range = adapter.context_manifest['schema']['object.body_mass']
            expected_mass = adapter.context[:, slice(*mass_range)].cpu().numpy()
            live_mass = mw.body_mass.numpy()[..., object_bodies]
            if not np.allclose(live_mass, expected_mass, atol=1e-8, rtol=1e-6):
                raise ValueError('MJWarp object mass differs from the recorded randomized context')
            mass_audit = dict(after_physics_step=True,
                max_error_kg=float(np.abs(live_mass-expected_mass).max()),
                minimum_mass_kg=float(live_mass.min()), maximum_mass_kg=float(live_mass.max()))
            timestep = mw.opt.timestep.numpy()
            expected_timestep = env.physics_dt / env.cfg.sim.physics.num_substeps
            if (not np.isfinite(timestep).all()
                    or not np.allclose(timestep, expected_timestep, atol=1e-9, rtol=1e-6)):
                raise ValueError('Live MJWarp timestep differs from the declared solver substeps')
            from viserdex.supermimic.physics_health import audit_compiled_joint_limits
            limits_audit = audit_compiled_joint_limits(
                NewtonManager._solver, REVO3_JOINT_NAMES, adapter.manual_asset_audit['limits_rad'])
            report['randomized_object_mass_audit'] = mass_audit
            report['post_first_physics_step_audit'] = dict(
                after_physics_step=True, preserve_reset_state=a.preserve_reset_state,
                pd_period_s=env.physics_dt, policy_period_s=env.step_dt,
                num_substeps=env.cfg.sim.physics.num_substeps,
                expected_solver_timestep_s=expected_timestep,
                actual_solver_timestep_min_s=float(timestep.min()),
                actual_solver_timestep_max_s=float(timestep.max()),
                compiled_joint_limits=limits_audit)
            atomic_json(report_path, report)

        class Trials:
            velocity = [0.]*6
            def reset(self, ids):
                ids = torch.arange(env.num_envs, device=env.device) if ids is None else ids
                write_grasp_states(adapter, ids, [self.rows[int(i)] for i in ids], velocity=self.velocity)
            def post_step(self, *args): pass
            def transition_ids(self): return {}

        driver = Trials()
        env.grasp_reset_manager = driver
        batch_size = env.num_envs // a.contexts
        for begin in range(0, len(records), batch_size):
            batch = records[begin:begin+batch_size]
            padded = batch + [batch[-1]]*(batch_size-len(batch))
            driver.rows = [dict(r) for r in padded for _ in range(a.contexts)]
            for r in driver.rows:
                validate_record(r, limits, certified=False, contract=contract)
            driver.velocity = [0.]*6
            env.reset(); synchronize_reset(adapter)
            target = adapter.state.new_tensor([r['hold_target_encoder'] for r in driver.rows])
            actions = 2*(target-adapter.limits[..., 0])/(adapter.limits[..., 1]-adapter.limits[..., 0])-1
            settling_done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            for _ in range(0 if a.preserve_reset_state else 100):
                transition, *_ = adapter.step(actions)
                audit_first_physics_step()
                settling_done |= transition['done']
            settled = adapter.physical().cpu().numpy()
            canonical = []
            for j, r in enumerate(padded):
                state = settled[j*a.contexts]
                r = (dict(r, q_geometry=state[:21].tolist(), object_pose_xyzw=state[42:49].tolist())
                     if not a.preserve_reset_state else dict(r))
                r['reset_state_origin'] = 'supplied_state_recertified' if a.preserve_reset_state else 'actual_5s_settled_state'
                pose = r['object_pose_xyzw']
                query.set_pose(r['q_geometry'], pose[:3], pose[3:])
                r['reset_min_gap_m'] = query.minimum_gap()
                r['reset_finger_gaps_m'] = [query.closest(g)[0] for g in range(5)]
                r['settle_done'] = bool(settling_done[j*a.contexts:(j+1)*a.contexts].any())
                r['proposal_id'] = r['id']
                r['id'] = f'certified-{a.seed}-' + r['id'].removeprefix('certified-')
                r['certification_contexts'] = adapter.context[j*a.contexts:(j+1)*a.contexts].cpu().tolist()
                canonical.append(r)
            driver.rows = [r for r in canonical for _ in range(a.contexts)]
            evidence = [[] for _ in canonical]
            eligible = [not r['settle_done'] and r['reset_min_gap_m'] >= -.0015 for r in canonical]
            print(json.dumps(dict(batch=begin, settled=[dict(id=r['id'], min_gap=r['reset_min_gap_m'], eligible=ok) for r,ok in zip(canonical, eligible)])), flush=True)
            for j, r in enumerate(canonical):
                try: validate_record(r, limits, certified=False, contract=contract)
                except ValueError: eligible[j] = False
            if any(eligible):
                for name, (steps, velocity) in trial_protocol().items():
                    driver.velocity = velocity
                    env.reset(); synchronize_reset(adapter)
                    reference = adapter.state[:, 42:49].clone()
                    peak = torch.zeros(env.num_envs, 4, device=env.device)
                    done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
                    contact_penetration = np.zeros(env.num_envs)
                    contact_samples = np.zeros(env.num_envs, dtype=int)
                    for _ in range(steps):
                        transition, *_ = adapter.step(actions)
                        audit_first_physics_step()
                        state = transition['next_physical']
                        metrics = torch.stack(((state[:, 42:45]-reference[:, :3]).norm(dim=-1),
                            rotation_log(quat_mul(state[:, 45:49], quat_conjugate(reference[:, 3:]))).norm(dim=-1),
                            state[:, 49:52].norm(dim=-1), state[:, 52:55].norm(dim=-1)), -1)
                        peak = torch.maximum(peak, metrics)
                        done |= transition['done']
                        nc = int(contact.nacon.numpy()[0])
                        geoms = contact.contact.geom.numpy()[:nc]
                        worlds = contact.contact.worldid.numpy()[:nc]
                        distances = contact.contact.dist.numpy()[:nc]
                        active = contact.contact.efc_address.numpy()[:nc, 0] >= 0
                        accumulate_object_contacts(geoms, worlds, distances, active, object_geoms,
                                                   contact_samples, contact_penetration)
                    for j in range(len(canonical)):
                        section = slice(j*a.contexts, (j+1)*a.contexts)
                        values = dict(zip(THRESHOLDS, peak[section].max(0).values.cpu().tolist()))
                        passed = (not bool(done[section].any()) and all(values[k] <= v for k,v in THRESHOLDS.items())
                                  and float(contact_penetration[section].max()) <= .0015
                                  and float((contact_samples[section]/steps).min()) >= .9)
                        evidence[j].append(dict(name=name, steps=steps, done=bool(done[section].any()), passed=passed,
                            solver_object_penetration_m=float(contact_penetration[section].max()),
                            object_contact_fraction=float((contact_samples[section]/steps).min()),
                            contexts=adapter.context[section].cpu().tolist(), velocity_base=velocity, **values))
                        eligible[j] &= passed
                    print(json.dumps(dict(batch=begin, trial=name, remaining=sum(eligible))), flush=True)
                    if not any(eligible):
                        break
            for j, r in enumerate(canonical[:len(batch)]):
                cert = dict(protocol=CERTIFICATE, contract_sha256=digest(contract), state_sha256=state_digest(r),
                            thresholds=THRESHOLDS, reset_min_gap_m=r['reset_min_gap_m'], reset_penetration_limit_m=.0015,
                            overflow=False, passed=bool(eligible[j]), trials=evidence[j])
                r['certificate'] = cert
                r['certified'] = bool(eligible[j])
                # A source group has one deterministic split, including all descendants.
                bucket = int(digest(r['source_group'])[:8], 16) % 10
                r['split'] = 'validation' if bucket == 0 else 'test' if bucket == 1 else 'train'
                if eligible[j]:
                    validate_record(r, limits, contract=contract)
                    accepted.append(r)
                report['results'].append(r)
            accepted = deduplicate(accepted, limits)
            report.update(accepted=len(accepted), contact_peaks=adapter.contact_peaks, max_errors=adapter.max_errors,
                          elapsed_seconds=time.monotonic()-start, overflow=False)
            atomic_json(report_path, report)
            atomic_json(a.output, dict(schema=SCHEMA, contract=contract, records=accepted))
        if not records:
            report.update(accepted=len(accepted), overflow=False)
            atomic_json(a.output, dict(schema=SCHEMA, contract=contract, records=accepted))
        report['complete'] = True
        atomic_json(report_path, report)
        print(json.dumps(dict(accepted=len(accepted), report=str(report_path), bank=str(a.output))), flush=True)
    finally:
        env.close()


def run_jobs():
    index, observed = 0, []
    started = time.monotonic()
    deadline = started + a.wait_timeout
    while True:
        if not a.jobs_manifest.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Timed out waiting for certification manifest')
            time.sleep(.2)
            continue
        data = json.loads(a.jobs_manifest.read_text())
        if data.get('schema') != 'revo3_certification_jobs_v1' or data.get('error'):
            raise ValueError('Invalid or failed certification manifest')
        jobs = data['jobs']
        if jobs[:index] != observed:
            raise ValueError('Certification manifest changed previously consumed jobs')
        if index == len(jobs):
            if data.get('complete'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Timed out waiting for next certification job')
            time.sleep(.2)
            continue
        job = jobs[index]
        a.candidates, a.output, a.seed = Path(job['candidates']), Path(job['output']), int(job['seed'])
        if a.output.exists() or a.output.with_suffix('.report.json').exists():
            raise FileExistsError('Refusing to overwrite certification: ' + str(a.output))
        job_started = time.monotonic()
        completed_job()
        observed.append(job)
        index += 1
        print(json.dumps(dict(completed_job=index, job_wall_seconds=time.monotonic()-job_started,
                              worker_elapsed_seconds=time.monotonic()-started)), flush=True)
        deadline = time.monotonic() + a.wait_timeout


if __name__ == '__main__':
    try:
        run_jobs() if a.jobs_manifest else completed_job()
    finally:
        app.close()
