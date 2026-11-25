"""Versioned physical grasp states and bounded adaptive start/goal sampling.

A bank contains no invented transitions. Certificates are bound to the exact
reset state, hold target, hand/object geometry and simulation contract.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'revo3_newton_grasp_bank_v1'
CERTIFICATE = 'newton_hold_reset_perturb_v1'
THRESHOLDS = dict(position_drift_m=.003, angle_drift_rad=np.deg2rad(5.),
                  linear_speed_m_s=.05, angular_speed_rad_s=.5)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def state_digest(record):
    return digest({k: record[k] for k in ('q_geometry', 'hold_target_encoder', 'object_pose_xyzw')})


def asset_contract(robot_path, object_path, spec, limits):
    from .identified_hand import load_identified_hand
    from .geometry import geometry_digest
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    from .sdk_limits import sampling_geometry_limits
    from .training_limits import load_training_limits
    from viserdex.assets.prepare_manual_convex import inspect_manual_asset
    parameters, identified = load_identified_hand()
    expected, source = load_training_limits()
    if not np.allclose(limits, expected, atol=2e-7, rtol=0):
        raise ValueError('Grasp limits differ from the independent training configuration')
    mechanical = inspect_manual_asset(robot_path)['limits_rad']
    geometry_limits = sampling_geometry_limits(expected, parameters['encoder_bias_rad'], mechanical)
    return dict(joint_names=list(REVO3_JOINT_NAMES), geometry_sha256=geometry_digest(spec),
                robot_sha256=hashlib.sha256(Path(robot_path).read_bytes()).hexdigest(),
                object_sha256=hashlib.sha256(Path(object_path).read_bytes()).hexdigest(),
                joint_limits_rad=np.asarray(limits).tolist(), quaternion_order='XYZW', frame='hand_base',
                sdk_software_limits=source['sdk_software_limits'], training_soft_limits=source,
                geometry_sampling_limits_rad=geometry_limits.tolist(),
                encoder_bias_rad=parameters['encoder_bias_rad'], mechanical_limits_rad=mechanical,
                identified_sha256=identified['sha256']['best_parameters.json'], control_dt=.05)


def live_contract(adapter):
    from importlib.metadata import version
    from viserdex.distillation.environment import contract_sha256
    env = adapter.env
    assets = asset_contract(env.cfg.scene.robot.spawn.usd_path, env.cfg.scene.object.spawn.usd_path,
                            adapter.geometry_spec, adapter.limits[0].cpu().numpy())
    contract = dict(assets=assets, context_manifest=adapter.context_manifest,
                environment_sha256=contract_sha256(env.cfg.to_dict()),
                physics_sha256=digest(env.cfg.sim.physics.to_dict()),
                gravity=list(env.cfg.sim.gravity), base_inverse=adapter.base_inverse[0].cpu().tolist(),
                versions={key:version(key) for key in ('isaacsim','isaaclab','newton','warp-lang')})
    return json.loads(json.dumps(contract, allow_nan=False))


def trial_protocol():
    return {'reset': (40, [0.]*6), 'hold': (100, [0.]*6), **{
        f'kick_{axis}_{sign}': (100, [sign*(.05 if axis < 3 else .5) if j == axis else 0. for j in range(6)])
        for axis in range(6) for sign in (-1, 1)}}


def validate_record(record, limits, certified=True, contract=None):
    arrays = {}
    for name, size in (('q_geometry',21),('hold_target_encoder',21),('object_pose_xyzw',7)):
        value = np.asarray(record.get(name), dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid grasp {name}')
        arrays[name] = value
    if abs(np.linalg.norm(arrays['object_pose_xyzw'][3:])-1) > 1e-4:
        raise ValueError('Grasp requires unit XYZW quaternion')
    assets = (contract or {}).get('assets', {})
    geometry_limits = np.asarray(assets.get('geometry_sampling_limits_rad', limits))
    for name, bounds in (('q_geometry', geometry_limits), ('hold_target_encoder', limits)):
        if ((arrays[name] < bounds[:,0]-1e-5) | (arrays[name] > bounds[:,1]+1e-5)).any():
            raise ValueError(f'Grasp {name} outside current coordinate limits')
    if not isinstance(record.get('id'), str) or not record['id']:
        raise ValueError('Grasp ID is required')
    if not record.get('source_group'):
        raise ValueError('Grasp ancestry/source group is required')
    if not certified:
        return
    cert = record.get('certificate', {})
    if cert.get('protocol') != CERTIFICATE or cert.get('state_sha256') != state_digest(record):
        raise ValueError('Grasp certificate does not bind its current state')
    trials = cert.get('trials', [])
    if contract is None or cert.get('contract_sha256') != digest(contract):
        raise ValueError('Grasp certificate dynamics contract mismatch')
    if cert.get('thresholds') != THRESHOLDS or cert.get('overflow') is not False:
        raise ValueError('Invalid certificate thresholds/capacity audit')
    gap = cert.get('reset_min_gap_m')
    if cert.get('reset_penetration_limit_m') != .0015 or not isinstance(gap, (int, float)) or not np.isfinite(gap) or gap < -.0015:
        raise ValueError('Certified reset state has excessive geometric penetration')
    expected = trial_protocol()
    if len(trials) != len(expected) or {t['name'] for t in trials} != set(expected):
        raise ValueError('Incomplete reset/hold/perturbation evidence')
    for trial in trials:
        if (trial.get('steps') != expected[trial['name']][0] or trial.get('done') is not False
                or not trial.get('contexts') or trial.get('passed') is not True):
            raise ValueError('Grasp did not complete every physical validation trial')
        contexts = np.asarray(trial['contexts'], dtype=float)
        if contexts.ndim != 2 or contexts.shape[1] != 65 or len(contexts) < 2 or not np.isfinite(contexts).all():
            raise ValueError('Certificate requires at least two actual xi65 contexts')
        if len(np.unique(contexts, axis=0)) < 2:
            raise ValueError('Certificate contexts must include randomization')
        if trial.get('velocity_base') != expected[trial['name']][1]:
            raise ValueError('Incorrect physical perturbation')
        penetration, support = trial.get('solver_object_penetration_m'), trial.get('object_contact_fraction')
        if (not isinstance(penetration, (float, int)) or not np.isfinite(penetration) or not 0 <= penetration <= .0015
                or not isinstance(support, (float, int)) or not np.isfinite(support) or not .9 <= support <= 1):
            raise ValueError('Missing or invalid actual solver contact evidence')
        for key, limit in THRESHOLDS.items():
            value = trial.get(key)
            if type(value) not in (int,float) or not np.isfinite(value) or value < 0 or value > limit:
                raise ValueError(f'Grasp certificate fails {key}')
    if cert.get('passed') is not True:
        raise ValueError('Uncertified grasp cannot enter a training bank')


def deduplicate(records, limits):
    """Greedy physical reset clusters; descendants retain their source group."""
    kept = []
    span = limits[:,1]-limits[:,0]
    for record in records:
        q, pose = np.asarray(record['q_geometry']), np.asarray(record['object_pose_xyzw'])
        duplicate = False
        for prior in kept:
            p = np.asarray(prior['object_pose_xyzw'])
            angle = 2*np.arccos(np.clip(abs(pose[3:]@p[3:]),0,1))
            if (np.sqrt(np.mean(((q-prior['q_geometry'])/span)**2)) < .02
                    and np.linalg.norm(pose[:3]-p[:3]) < .002 and angle < np.deg2rad(5)):
                duplicate = True
                break
        if not duplicate:
            kept.append(record)
    return kept


class GraspBank:
    def __init__(self, path, contract, split='train'):
        self.path = str(Path(path).resolve())
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if data.get('schema') != SCHEMA or data.get('contract') != contract:
            raise ValueError('Grasp bank asset/dynamics contract mismatch; recertify in this environment')
        limits = np.asarray(contract['assets']['joint_limits_rad'])
        records = data.get('records', [])
        if not records:
            raise ValueError('Grasp bank contains no certified states')
        for r in records:
            validate_record(r, limits, contract=contract)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Duplicate grasp IDs')
        groups = {}
        for r in records:
            if r.get('split') not in ('train','validation','test'):
                raise ValueError('Invalid grasp split')
            if r['source_group'] in groups and groups[r['source_group']] != r['split']:
                raise ValueError('Grasp source group leaks across data splits')
            groups[r['source_group']] = r['split']
        self.records = [r for r in records if r['split'] == split]
        if not self.records:
            raise ValueError(f'No certified grasps in split {split}')
        self.split = split
        self.pose = np.asarray([r['object_pose_xyzw'] for r in self.records])


class PairSampler:
    """Difficulty bins with bounded learning-progress weights and uniform floor.

    A sampled pair is one fixed command for an entire episode. Any bank member
    can be a segment start/end, so waypoint curricula need no hidden goal switch.
    """
    def __init__(self, records, seed=0, adaptive=True):
        self.rng = np.random.default_rng(seed)
        self.adaptive = adaptive
        self.records = records
        self.n = len(records)
        if not self.n:
            raise ValueError('Pair sampler needs records')
        q = np.asarray([r['object_pose_xyzw'][3:] for r in records])
        angles = 2*np.arccos(np.clip(np.abs(q@q.T),0,1))
        self.bins = np.digitize(angles, np.deg2rad([15,45,90,135]))
        self.count = np.zeros(5,dtype=int)
        self.success = np.full(5,.5)
        self.groups = {}
        for i,r in enumerate(records): self.groups.setdefault(r['source_group'],[]).append(i)
        self.group_members = list(self.groups.values())

    def sample(self, n):
        starts, goals, buckets = [], [], []
        for _ in range(n):
            group = self.group_members[self.rng.integers(len(self.group_members))]
            start = int(self.rng.choice(group))
            available = np.bincount(self.bins[start],minlength=5)>0
            weight = np.ones(5)
            if self.adaptive:
                weight = (.1+4*self.success*(1-self.success))/np.sqrt(1+self.count/20)
            weight *= available
            # The floor is uniform over actual goals, not over unequal bins.
            uniform = np.bincount(self.bins[start], minlength=5) / self.n
            p = .2*uniform+.8*weight/weight.sum() if self.adaptive else uniform
            bucket = int(self.rng.choice(5,p=p))
            goal = int(self.rng.choice(np.flatnonzero(self.bins[start]==bucket)))
            starts.append(start);goals.append(goal);buckets.append(bucket)
        return np.array(starts),np.array(goals),np.array(buckets)

    def update(self, buckets, successes):
        for b, ok in zip(buckets, successes):
            self.success[b] = .95*self.success[b]+.05*float(ok)
            self.count[b] += 1

    def state_dict(self):
        return dict(count=self.count.tolist(),success=self.success.tolist(),rng=self.rng.bit_generator.state)

    def load_state_dict(self, data):
        count, success = np.asarray(data['count']), np.asarray(data['success'], dtype=float)
        if count.shape != (5,) or success.shape != (5,) or not np.isfinite(count).all() or not np.isfinite(success).all() or (count < 0).any() or (count != count.astype(int)).any() or ((success < 0) | (success > 1)).any():
            raise ValueError('Invalid adaptive grasp statistics')
        self.count, self.success = count.astype(int), success
        self.rng.bit_generator.state=data['rng']
