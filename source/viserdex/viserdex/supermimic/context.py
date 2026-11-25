"""Lossless startup domain context: variable fields, plus fixed reconstruction metadata.

Selection follows configured ranges, never sample variance (including one-env runs).
Fixed hand contact friction is metadata; randomized object friction stays in xi.
"""
import torch
from .identified_hand import load_identified_hand

FIXED_PALM_BODY = 'world'
MOVING_HAND_BODIES_PATTERN = r'^(?!world$).+'


def env_identified_profile(env):
    cfg = getattr(env, 'cfg', None)
    if cfg is None:
        return 'fit_03'
    actions = getattr(cfg, 'actions', None)
    joint_pos = getattr(actions, 'joint_pos', None) if actions is not None else None
    dex_profile = getattr(joint_pos, 'dex_reference_profile', None)
    if isinstance(dex_profile, dict) and dex_profile.get('identified_hand_profile'):
        return dex_profile['identified_hand_profile']
    return 'fit_03'


def moving_hand_body_indices(body_names, is_fixed_base):
    """This audited asset's palm is the world-fixed articulation root."""
    if not is_fixed_base or list(body_names).count(FIXED_PALM_BODY) != 1:
        raise ValueError('Palm mass omission requires the audited fixed-base world body')
    if len(body_names) != 22 or len(set(body_names)) != 22:
        raise ValueError('Expected one fixed palm and 21 moving links')
    return [i for i, name in enumerate(body_names) if name != FIXED_PALM_BODY]


def pack_context(fields, variable):
    chunks, schema, fixed, start = [], {}, {}, 0
    for name, value in fields.items():
        value = value.reshape(len(value), -1).clone()
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite context: {name}')
        if variable[name]:
            chunks.append(value)
            schema[name] = [start, start + value.shape[-1]]
            start += value.shape[-1]
        else:
            torch.testing.assert_close(value, value[:1].expand_as(value), atol=1e-8, rtol=1e-6)
            fixed[name] = value[0].cpu().tolist()
    if chunks:
        return torch.cat(chunks, -1), schema, fixed
    reference = next(iter(fields.values()))
    return reference.new_empty(len(reference), 0), schema, fixed


def compact_context(env, fields):
    fields = dict(fields)
    variable, derived, ranges, excluded = {}, {}, {}, {}
    def event(name):
        cfg = env.event_manager.get_term_cfg(name)
        if cfg.mode != 'startup':
            raise ValueError('Replay context requires startup-only domain randomization')
        return cfg
    def varying(bounds):
        return bounds[0] != bounds[1]
    for name in ('robot', 'object'):
        mass_event = event(name + '_scale_mass')
        params = mass_event.params
        if params['operation'] != 'scale' or not params.get('recompute_inertia', True):
            raise ValueError('Compact inertia reconstruction requires mass scaling with recomputed inertia')
        ranges[name + '.mass_scale'] = list(params['mass_distribution_params'])
        variable[name + '.body_mass'] = varying(params['mass_distribution_params'])
        default_mass = mass_event.func.default_mass.to(env.device)
        default_inertia = mass_event.func.default_inertia.to(env.device)
        mass = fields[name + '.body_mass']
        inertia = fields.pop(name + '.body_inertia')
        body_names = list(mass_event.func.asset.body_names)
        if name == 'robot':
            keep = moving_hand_body_indices(body_names, mass_event.func.asset.is_fixed_base)
            palm = body_names.index(FIXED_PALM_BODY)
            # Check the selector reached the real randomizer, not just metadata.
            selected = list(range(len(body_names)))[params['asset_cfg'].body_ids] if isinstance(
                params['asset_cfg'].body_ids, slice) else list(params['asset_cfg'].body_ids)
            if set(int(i) for i in selected) != set(keep):
                raise ValueError('Hand mass randomization must select only the 21 moving bodies')
            torch.testing.assert_close(mass[:, palm], default_mass[:, palm], atol=0, rtol=0)
            torch.testing.assert_close(inertia[:, palm], default_inertia[:, palm], atol=0, rtol=0)
            mass, inertia = mass[:, keep], inertia[:, keep]
            default_mass, default_inertia = default_mass[:, keep], default_inertia[:, keep]
            fields[name + '.body_mass'] = mass
            body_names = [body_names[i] for i in keep]
            excluded[name] = dict(mass_and_inertia=[FIXED_PALM_BODY], reason='world-fixed articulation root')
        if not (default_mass > 0).all():
            raise ValueError('Inertia reconstruction requires positive authored body mass')
        expected = default_inertia * (mass / default_mass)[..., None]
        torch.testing.assert_close(inertia, expected, atol=1e-10, rtol=2e-5)
        torch.testing.assert_close(default_mass, default_mass[:1].expand_as(default_mass))
        torch.testing.assert_close(default_inertia, default_inertia[:1].expand_as(default_inertia))
        derived[name + '.body_inertia'] = dict(
            formula='default_inertia * live_mass / default_mass',
            body_names=body_names,
            default_mass=default_mass[0].cpu().tolist(),
            default_inertia=default_inertia[0].cpu().tolist())
        material = event(name + '_physics_material').params
        for field, parameter in (('shape_material_mu', 'static_friction_range'),
                                 ('shape_material_restitution', 'restitution_range')):
            variable[name + '.' + field] = varying(material[parameter])
            ranges[name + '.' + field] = list(material[parameter])
            if not variable[name + '.' + field]:
                value = fields[name + '.' + field]
                torch.testing.assert_close(value, torch.full_like(value, material[parameter][0]), atol=1e-7, rtol=1e-6)
    gain = event('robot_joint_stiffness_and_damping').params
    if gain['operation'] != 'scale' or gain['distribution'] != 'log_uniform':
        raise ValueError('Identified gain context expects configured log-uniform scale randomization')
    for name in fields:
        if name.startswith('actuator.'):
            parameter = 'stiffness_distribution_params' if name.endswith('.kp') else 'damping_distribution_params'
            variable[name] = varying(gain[parameter])
            ranges[name] = list(gain[parameter])
    variable['joint_limits'] = False
    context, schema, fixed = pack_context(fields, variable)
    p, info = load_identified_hand(env_identified_profile(env))
    # Fixed identified parameters remain reproducible without redundant network inputs.
    manifest = dict(protocol='configured_random_moving_bodies_v2', fixed=fixed,
                                 derived=derived, ranges=ranges, excluded_bodies=excluded, identified_fixed=p,
                                 identified_source=info, schema=schema)
    env_cfg = getattr(env, 'cfg', None)
    dex_profile = getattr(getattr(getattr(env_cfg, 'actions', None), 'joint_pos', None), 'dex_reference_profile', None)
    if dex_profile is not None:
        manifest['identified_fixed'] = dict(p, delay_s=0., encoder_bias_rad=[0.]*21)
        manifest['control_override'] = dict(protocol='v33', delay_steps=0, fixed_encoder_compensation=False,
                                           profile=dex_profile)
    from .output_limits import environment_output_limits
    limits = environment_output_limits(env)
    if limits is not None:
        manifest['actuator_output_limit_override'] = limits
    from .runtime_capacity import capacity_manifest
    capacity = capacity_manifest()
    if capacity is not None:
        manifest['mjwarp_capacity_override'] = capacity
    return context, schema, manifest


def policy_order_parameter(robot, name, component, attr, joint_names=None):
    from isaaclab.actuators.newton import read_group_parameter
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    names = robot.actuators._group_joint_names[name]
    order = [names.index(joint) for joint in (joint_names or REVO3_JOINT_NAMES)]
    return read_group_parameter(robot.actuators, name, component, attr)[:, order]


def audit_delay_memory(ready):
    """Prove one last command plus reset flag reconstructs all pending commands."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    adapter = NewtonManager._adapter
    checked = 0
    for actuator, state in zip(adapter.actuators, adapter._states_a):
        if state is None or state.delay_state is None:
            continue
        env_ids = wp.to_torch(actuator.indices).long() // adapter.num_joints
        active = ready.flatten()[env_ids].bool()
        delay = state.delay_state
        pushes = wp.to_torch(delay.num_pushes)
        torch.testing.assert_close(pushes, active.to(pushes.dtype) * actuator.delay.buf_depth)
        if active.any():
            target = wp.to_torch(NewtonManager.backend.control.joint_target_q)
            target = target[wp.to_torch(actuator.target_pos_indices).long()][active]
            pending = wp.to_torch(delay.buffer_pos)[:, active]
            torch.testing.assert_close(pending, target[None].expand_as(pending), atol=1e-7, rtol=1e-6)
        for field in (delay.buffer_vel, delay.buffer_act):
            values = wp.to_torch(field)
            torch.testing.assert_close(values, torch.zeros_like(values), atol=0, rtol=0)
        checked += len(pushes)
    if checked != len(ready) * 21:
        raise ValueError('Delay audit did not cover exactly 21 hand actuators per environment')


def audit_identified_actuator(env, robot, ids, fields, joint_names=None):
    from isaaclab.actuators.newton import read_group_parameter
    p, info = load_identified_hand(env_identified_profile(env))
    from .output_limits import environment_output_limits
    limits = environment_output_limits(env)
    if joint_names is not None:
        from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
        order = [REVO3_JOINT_NAMES.index(name) for name in joint_names]
        p = {key: [value[i] for i in order] if isinstance(value, list) else value for key, value in p.items()}
    if abs(env.physics_dt - info['pd_period_s']) > 1e-9 or env.cfg.decimation not in (8, 10) or info['delay_steps'] >= env.cfg.decimation:
        raise ValueError('PACE requires 5 ms PD ticks, 20/25 Hz policy commands, delay below one command period')
    from isaaclab_newton.physics import NewtonManager
    substep_record = None
    if NewtonManager._num_substeps != info['physics_substeps']:
        from .solver_budget import digest, validate_solver_override, substep_experiment_record
        proof = getattr(env, '_supermimic_solver_override_proof', None)
        if proof is None or proof.get('substeps_experiment') is not True:
            raise ValueError('PACE physics substeps changed without an explicit numerical experiment')
        validate_solver_override(proof, digest(proof['source']), digest(proof['target']))
        if proof['target']['sim']['physics']['num_substeps'] != NewtonManager._num_substeps:
            raise ValueError('Live solver substeps differ from the declared experiment')
        substep_record = substep_experiment_record(info, NewtonManager._num_substeps)
    for attr, field in (('joint_armature', 'armature_kg_m2'),
                        ('joint_friction_coeff', 'friction_Nm'),
                        ('joint_viscous_friction_coeff', 'viscous_Nm_s_rad')):
        actual = getattr(robot.data, attr).torch[:, ids]
        torch.testing.assert_close(actual, actual.new_tensor(p[field])[None].expand_as(actual), atol=1e-8, rtol=2e-6)
    for component, attr, expected in (
        ('clamping', 'max_effort', p['effective_effort_limit_Nm'] if limits is None
            else [limits['effort_limit_Nm']] * 21),
        ('delay', 'delay_steps', [0 if getattr(env.cfg.actions.joint_pos, 'dex_reference_profile', None) is not None
                                   else info['delay_steps']] * 21)):
        if component == 'delay' and getattr(env.cfg.actions.joint_pos, 'dex_reference_profile', None) is not None:
            # Newton omits the delay component entirely when both configured delays are zero.
            adapter = NewtonManager._adapter
            covered = 0
            for actuator, state in zip(adapter.actuators, adapter._states_a):
                if getattr(actuator, 'delay', None) is not None or (state is not None and state.delay_state is not None):
                    raise ValueError('v33 requires absent live actuator delay and queue state')
                covered += len(actuator.indices)
            if covered != env.num_envs*21:
                raise ValueError('No-delay audit must cover all 21 actuators per environment')
            continue
        actual = policy_order_parameter(robot, 'fingers', component, attr, joint_names)
        torch.testing.assert_close(actual, actual.new_tensor(expected)[None].expand_as(actual), atol=1e-7, rtol=2e-6)
    gains = env.event_manager.get_term_cfg('robot_joint_stiffness_and_damping').params
    for short, field, parameter in (('kp', 'effective_kp_Nm_rad', 'stiffness_distribution_params'),
                                     ('kd', 'effective_kd_Nm_s_rad', 'damping_distribution_params')):
        actual = fields['actuator.fingers.' + short]
        ratio = actual / actual.new_tensor(p[field])
        lo, hi = gains[parameter]
        if not ((ratio >= lo - 1e-6) & (ratio <= hi + 1e-6)).all():
            raise ValueError(f'Live {short} is not randomized around the identified baseline')
    result = dict(info, live_parameters_verified=True, encoder_bias_rad=p['encoder_bias_rad'])
    if getattr(env.cfg.actions.joint_pos, 'dex_reference_profile', None) is not None:
        result.update(delay_steps=0, effective_delay_s=0., encoder_bias_rad=[0.]*21,
                      encoder_zero_randomization_deg=env.cfg.actions.joint_pos.dex_reference_profile['zero_bias_deg'])
    if limits is not None:
        for attr, key in (('joint_effort_limits','effort_limit_Nm'),
                          ('joint_vel_limits','velocity_limit_rad_s')):
            actual = getattr(robot.data, attr).torch[:, ids]
            torch.testing.assert_close(actual, torch.full_like(actual, limits[key]), atol=2e-6, rtol=2e-6)
        result['actuator_output_limit_override'] = limits
    if substep_record is not None:
        result['solver_substep_experiment'] = substep_record
    return result
