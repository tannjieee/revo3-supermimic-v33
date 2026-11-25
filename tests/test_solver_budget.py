import copy
import pytest

from viserdex.supermimic.solver_budget import (
    digest, validate_iteration_override, validate_solver_override,
    substep_experiment_record, validate_identified_substep_override,
)


def contracts():
    source = {'sim': {'physics': {
        'class_type': 'isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager',
        'solver_cfg': {'iterations': 100, 'tolerance': 1e-6}}},
        'rewards': {'scale': 10}, 'scene': {'asset': 'original.usd'}}
    target = copy.deepcopy(source)
    target['sim']['physics']['solver_cfg']['iterations'] = 200
    return source, target


def test_iteration_budget_increase_and_parent_hash():
    source, target = contracts()
    proof = dict(source=source, target=target)
    assert validate_iteration_override(proof, digest(source), digest(target))['current'] == 200
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_iteration_override(proof, 'wrong checkpoint', digest(target))


@pytest.mark.parametrize('change', ['tolerance', 'reward', 'asset', 'decrease', 'equal', 'backend'])
def test_iteration_override_rejects_other_changes(change):
    source, target = contracts()
    if change == 'tolerance':
        target['sim']['physics']['solver_cfg']['tolerance'] = 1e-4
    elif change == 'reward':
        target['rewards']['scale'] = 1
    elif change == 'asset':
        target['scene']['asset'] = 'other.usd'
    elif change in ('decrease', 'equal'):
        target['sim']['physics']['solver_cfg']['iterations'] = 50 if change == 'decrease' else 100
    else:
        source['sim']['physics']['class_type'] = 'other'
        target['sim']['physics']['class_type'] = 'other'
    with pytest.raises(ValueError):
        validate_iteration_override(dict(source=source, target=target), digest(source), digest(target))


def substep_proof(old=2, new=4):
    source, _ = contracts()
    source['sim']['dt'] = .005
    source['decimation'] = 10
    source['sim']['physics']['num_substeps'] = old
    target = copy.deepcopy(source)
    target['sim']['physics']['num_substeps'] = new
    return dict(source=source, target=target, substeps_experiment=True)


@pytest.mark.parametrize('old,new', [(2, 4), (2, 8), (4, 8), (4, 4)])
def test_declared_substeps_preserve_control_timing(old, new):
    proof = substep_proof(old, new)
    result = validate_solver_override(proof, digest(proof['source']), digest(proof['target']))
    assert result['changes']['num_substeps'] == dict(previous=old, current=new)


@pytest.mark.parametrize('change', ['undeclared', 'decrease', 'fractional', 'nonmultiple', 'dt', 'decimation', 'reward'])
def test_substeps_cannot_smuggle_other_changes(change):
    proof = substep_proof()
    if change == 'undeclared':
        proof.pop('substeps_experiment')
    elif change in ('decrease', 'fractional', 'nonmultiple'):
        proof['target']['sim']['physics']['num_substeps'] = {'decrease': 1, 'fractional': 4., 'nonmultiple': 3}[change]
    elif change == 'dt':
        proof['target']['sim']['dt'] = .0025
    elif change == 'decimation':
        proof['target']['decimation'] = 20
    else:
        proof['target']['rewards']['scale'] = 1
    with pytest.raises(ValueError):
        validate_solver_override(proof, digest(proof['source']), digest(proof['target']))


def test_identified_metadata_override_is_exact_and_supports_later_resumes():
    source = dict(physics_substeps=2, pd_period_s=.005, delay_steps=3, effective_delay_s=.015,
                  encoder_bias_rad=[.001], live_parameters_verified=True)
    target = dict(source, solver_substep_experiment=substep_experiment_record(source, 4))
    validate_identified_substep_override(source, target, substep_proof())
    validate_identified_substep_override(target, target, substep_proof(4, 4))
    changed = copy.deepcopy(target)
    changed['encoder_bias_rad'][0] += .001
    with pytest.raises(ValueError, match='plant parameters'):
        validate_identified_substep_override(source, changed, substep_proof())
    changed = copy.deepcopy(target)
    changed['solver_substep_experiment']['effective_delay_s'] = .0075
    with pytest.raises(ValueError, match='annotation'):
        validate_identified_substep_override(source, changed, substep_proof())


def test_legacy_iteration_override_remains_strict_about_substeps():
    proof = substep_proof()
    proof['target']['sim']['physics']['solver_cfg']['iterations'] = 200
    with pytest.raises(ValueError, match='other environment'):
        validate_iteration_override(proof, digest(proof['source']), digest(proof['target']))
