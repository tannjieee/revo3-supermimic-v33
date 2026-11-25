"""Explicit, narrowly checked Newton numerical-budget experiments on resume."""
import copy
import hashlib
import json

from viserdex.distillation.environment import SavedEnvironmentBinding, environment_contract, contract_sha256


def digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_iteration_override(proof, source_digest, target_digest):
    source, target = proof['source'], proof['target']
    if digest(source) != source_digest or digest(target) != target_digest:
        raise ValueError('Solver budget override contract hash mismatch')
    old = source['sim']['physics']['solver_cfg']['iterations']
    new = target['sim']['physics']['solver_cfg']['iterations']
    if type(old) is not int or type(new) is not int or not 0 < old < new:
        raise ValueError('Solver budget override must strictly increase iterations')
    if source['sim']['physics']['class_type'] != 'isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager':
        raise ValueError('Solver budget override requires Newton MJWarp')
    expected = copy.deepcopy(source)
    expected['sim']['physics']['solver_cfg']['iterations'] = new
    if target != expected:
        raise ValueError('Solver budget override changed other environment settings')
    return dict(previous=old, current=new, source_contract_sha256=source_digest,
                target_contract_sha256=target_digest)


def validate_solver_override(proof, source_digest, target_digest):
    """Prove that only an increased iteration cap and/or declared substeps differ."""
    source, target = proof['source'], proof['target']
    if digest(source) != source_digest or digest(target) != target_digest:
        raise ValueError('Solver budget override contract hash mismatch')
    physics = source['sim']['physics']
    if physics['class_type'] != 'isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager':
        raise ValueError('Solver budget override requires Newton MJWarp')
    expected = copy.deepcopy(source)
    old_iterations = physics['solver_cfg']['iterations']
    new_iterations = target['sim']['physics']['solver_cfg']['iterations']
    if type(old_iterations) is not int or type(new_iterations) is not int or not 0 < old_iterations <= new_iterations:
        raise ValueError('Solver budget override cannot decrease iterations')
    expected['sim']['physics']['solver_cfg']['iterations'] = new_iterations
    changes = {}
    if old_iterations != new_iterations:
        changes['iterations'] = dict(previous=old_iterations, current=new_iterations)
    declared_substeps = proof.get('substeps_experiment', False)
    if type(declared_substeps) is not bool:
        raise ValueError('Substeps experiment must be explicitly declared')
    if declared_substeps:
        old = physics['num_substeps']
        new = target['sim']['physics']['num_substeps']
        if type(old) is not int or type(new) is not int or old < 2 or old % 2 or new < old or new % old:
            raise ValueError('Substeps experiment must preserve or multiply the identified solver subdivisions')
        if source['sim']['dt'] != .005 or source['decimation'] != 10:
            raise ValueError('Substeps experiment requires fixed 5 ms PD and 20 Hz policy periods')
        expected['sim']['physics']['num_substeps'] = new
        changes['num_substeps'] = dict(previous=old, current=new)
    if not changes:
        raise ValueError('Solver budget override did not request a numerical experiment')
    if target != expected:
        raise ValueError('Solver budget override changed other environment settings')
    return dict(previous=old_iterations, current=new_iterations, changes=changes,
                source_contract_sha256=source_digest, target_contract_sha256=target_digest)


def substep_experiment_record(identified, runtime_substeps):
    fit = identified['physics_substeps']
    if (type(runtime_substeps) is not int or runtime_substeps < fit or runtime_substeps % fit
            or fit != 2 or identified['pd_period_s'] != .005 or identified['delay_steps'] != 3
            or identified['effective_delay_s'] != .015):
        raise ValueError('Substeps experiment changed the identified actuator timing contract')
    return dict(fit_substeps=fit, runtime_substeps=runtime_substeps,
                pd_period_s=.005, effective_delay_s=.015, policy_period_s=.05)


def validate_identified_substep_override(source, target, proof):
    """Allow exactly the documented runtime-substep annotation, no plant changes."""
    validate_solver_override(proof, digest(proof['source']), digest(proof['target']))
    if proof.get('substeps_experiment') is not True:
        raise ValueError('Identified hand change requires a declared substeps experiment')
    expected_source, expected_target = copy.deepcopy(source), copy.deepcopy(target)
    source_record = expected_source.pop('solver_substep_experiment', None)
    target_record = expected_target.pop('solver_substep_experiment', None)
    if expected_source != expected_target:
        raise ValueError('Substeps experiment changed identified plant parameters')
    for contract, metadata, record in ((proof['source'], source, source_record),
                                        (proof['target'], target, target_record)):
        runtime = contract['sim']['physics']['num_substeps']
        expected = None if runtime == metadata['physics_substeps'] else substep_experiment_record(metadata, runtime)
        if record != expected:
            raise ValueError('Identified hand substeps annotation does not match the environment proof')
    return target_record


class SolverBudgetBinding(SavedEnvironmentBinding):
    def __init__(self, path, iterations=None, num_substeps=None):
        super().__init__(path)
        original = copy.deepcopy(self.contract)
        source_digest = self.contract_sha
        if iterations is not None:
            self.saved['sim']['physics']['solver_cfg']['iterations'] = iterations
        if num_substeps is not None:
            self.saved['sim']['physics']['num_substeps'] = num_substeps
        self.contract = environment_contract(self.saved)
        self.contract_sha = contract_sha256(self.saved)
        self.override_proof = dict(source=original, target=copy.deepcopy(self.contract))
        if num_substeps is not None:
            self.override_proof['substeps_experiment'] = True
        self.override_record = validate_solver_override(self.override_proof, source_digest, self.contract_sha)

    def verify(self, cfg):
        record = super().verify(cfg)
        record['solver_iteration_override'] = self.override_record
        if self.override_proof.get('substeps_experiment'):
            record['solver_substep_experiment'] = self.override_record['changes']['num_substeps']
        return record
