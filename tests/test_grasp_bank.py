import copy
import json
import numpy as np
import pytest
from viserdex.supermimic.grasp_bank import (SCHEMA, CERTIFICATE, THRESHOLDS, GraspBank, PairSampler,
    validate_record, state_digest, digest, trial_protocol)


def fixture_record():
    limits = np.tile([-2., 2.], (21, 1))
    contract = {'assets': {'joint_limits_rad': limits.tolist()}}
    r = dict(id='a', source_group='source-a', split='train', q_geometry=[0.]*21,
             hold_target_encoder=[.01]*21, object_pose_xyzw=[.05, 0., .12, 0., 0., 0., 1.])
    trials = [dict(name=n, steps=s, velocity_base=v, contexts=[[1.]*65, [2.]*65],
                   done=False, passed=True, solver_object_penetration_m=.0004, object_contact_fraction=1., **{k: 0. for k in THRESHOLDS})
              for n, (s,v) in trial_protocol().items()]
    r['certificate'] = dict(protocol=CERTIFICATE, state_sha256=state_digest(r), contract_sha256=digest(contract),
                            trials=trials, thresholds=THRESHOLDS, reset_min_gap_m=-.0004, reset_penetration_limit_m=.0015, overflow=False, passed=True)
    return r, limits, contract


def test_full_protocol_and_binding():
    r, limits, contract = fixture_record()
    assert len(r['certificate']['trials']) == 14
    validate_record(r, limits, contract=contract)
    for change in ('state', 'context', 'missing', 'overflow', 'contract', 'velocity', 'metric', 'done', 'contacts', 'geometry'):
        bad = copy.deepcopy(r)
        c = bad['certificate']
        if change == 'state': bad['q_geometry'][0] += .001
        if change == 'context': c['trials'][0]['contexts'][0].pop()
        if change == 'missing': c['trials'].pop()
        if change == 'overflow': c['overflow'] = True
        if change == 'contract': c['contract_sha256'] = 'wrong'
        if change == 'velocity': c['trials'][-1]['velocity_base'] = [0.]*6
        if change == 'metric': c['trials'][0]['position_drift_m'] = .00301
        if change == 'done': c['trials'][0]['done'] = True
        if change == 'contacts': c['trials'][0]['object_contact_fraction'] = .8
        if change == 'geometry': c['reset_min_gap_m'] = -.002
        with pytest.raises(ValueError): validate_record(bad, limits, contract=contract)


def test_reject_ancestry_leak_duplicate_and_empty_bank(tmp_path):
    r, _, contract = fixture_record()
    path = tmp_path/'bank.json'
    def load(rows):
        path.write_text(json.dumps(dict(schema=SCHEMA, contract=contract, records=rows)))
        return GraspBank(path, contract)
    assert len(load([r]).records) == 1
    for rows in ([], [r, r], [r, dict(r, id='b', split='test')]):
        with pytest.raises(ValueError): load(rows)


def test_sampler_uniform_floor_source_groups_and_restore():
    r, _, _ = fixture_record()
    rows = [dict(r, id=str(i), source_group='large' if i < 9 else 'small') for i in range(10)]
    rows[-1]['object_pose_xyzw'] = [.05, 0., .12, 0., 0., 1., 0.]
    sampler = PairSampler(rows, seed=10, adaptive=True)
    sampler.update([0]*100, [True]*100)
    start, goal, bucket = sampler.sample(4000)
    assert .46 < np.mean(start == 9) < .54  # source families have equal mass
    assert (bucket == 4).any()
    saved = sampler.state_dict()
    restored = PairSampler(rows, seed=999, adaptive=True)
    restored.load_state_dict(saved)
    for left, right in zip(sampler.sample(100), restored.sample(100)):
        np.testing.assert_equal(left, right)
    uniform = PairSampler(rows, seed=42, adaptive=False)
    assert .07 < np.mean(uniform.sample(5000)[1] == 9) < .13
