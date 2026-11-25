"""Serialized COM boundary shared by fresh environment and policy contracts."""
import hashlib
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / 'configs/supermimic_v33/position_boundary.yaml'
FIELDS = ('drop_boundary_mode', 'drop_ellipsoid_center_m', 'drop_ellipsoid_semi_axes_m',
          'drop_object_com_local_m', 'drop_boundary_sha256')


def _vector(value, name, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != 3 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            and (not positive or x > 0) for x in value):
        raise ValueError(f'{name} must contain three finite {"positive " if positive else ""}numbers')
    return tuple(float(x) for x in value)


def validate_boundary(cfg):
    if cfg.drop_boundary_mode not in ('root_sphere', 'com_ellipsoid'):
        raise ValueError('Unknown drop boundary mode')
    _vector(cfg.drop_ellipsoid_center_m, 'drop_ellipsoid_center_m')
    _vector(cfg.drop_ellipsoid_semi_axes_m, 'drop_ellipsoid_semi_axes_m', True)
    _vector(cfg.drop_object_com_local_m, 'drop_object_com_local_m')
    if cfg.drop_boundary_mode == 'com_ellipsoid' and cfg.network_revision != 'hardware_center25_v3':
        raise ValueError('COM ellipsoid requires the hand-base compact policy contract')


def load_boundary(path=DEFAULT_CONFIG):
    import yaml
    raw = Path(path).read_bytes()
    p = yaml.safe_load(raw)
    required = dict(schema='revo3_object_com_ellipsoid_v1', behavior='hard_termination',
        object_variant='die58', shape='ellipsoid', coordinate_frame='hand_base',
        position_reference='object_center_of_mass', units='m', axis_alignment='hand_base_xyz')
    if not isinstance(p, dict) or any(p.get(k) != v for k, v in required.items()) or p.get('threshold_rho') != 1.0:
        raise ValueError('Expected the die58 hand-base COM hard ellipsoid with threshold rho=1')
    return dict(center=_vector(p.get('center_m'), 'center_m'),
        semi_axes=_vector(p.get('semi_axes_m'), 'semi_axes_m', True),
        boundary_sha256=hashlib.sha256(raw).hexdigest())


def configure_fresh_boundary(env_cfg, path=DEFAULT_CONFIG):
    from isaaclab.managers import TerminationTermCfg
    from pxr import Usd, UsdGeom, UsdPhysics
    from .object_variant import object_metadata
    from .spatial_contract import object_outside_com_ellipsoid
    asset = env_cfg.scene.object.spawn
    if object_metadata(asset.usd_path)['variant'] != 'die58':
        raise ValueError('Selected position boundary requires the 58 mm die')
    if asset.scale is not None and tuple(asset.scale) != (1., 1., 1.):
        raise ValueError('COM boundary requires the unscaled SI object asset')
    stage = Usd.Stage.Open(str(asset.usd_path))
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid) != 1 or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('COM boundary requires one SI rigid object')
    params = load_boundary(path)
    params['object_com_local'] = _vector(list(UsdPhysics.MassAPI(rigid[0]).GetCenterOfMassAttr().Get()), 'asset COM')
    env_cfg.terminations.object_out_of_reach = TerminationTermCfg(
        func=object_outside_com_ellipsoid, params=params)


def prepare_saved_boundary(env_cfg, saved):
    """Seed the typed params before SavedEnvironmentBinding.from_dict."""
    from .spatial_contract import object_outside_com_ellipsoid
    term = saved.get('terminations', {}).get('object_out_of_reach') or {}
    if term.get('func') == 'viserdex.supermimic.spatial_contract:object_outside_com_ellipsoid':
        from isaaclab.managers import TerminationTermCfg
        env_cfg.terminations.object_out_of_reach = TerminationTermCfg(
            func=object_outside_com_ellipsoid, params=dict(term['params']))


def policy_boundary_options(env_cfg):
    from .spatial_contract import object_outside_com_ellipsoid
    term = getattr(env_cfg.terminations, 'object_out_of_reach', None)
    if term is None or term.func not in (object_outside_com_ellipsoid,
                        'viserdex.supermimic.spatial_contract:object_outside_com_ellipsoid'):
        return {}
    p = term.params
    return dict(drop_boundary_mode='com_ellipsoid',
        drop_ellipsoid_center_m=_vector(p['center'], 'center'),
        drop_ellipsoid_semi_axes_m=_vector(p['semi_axes'], 'semi_axes', True),
        drop_object_com_local_m=_vector(p['object_com_local'], 'object_com_local'),
        drop_boundary_sha256=p['boundary_sha256'])


def verify_policy_boundary(cfg, env_cfg):
    expected = policy_boundary_options(env_cfg)
    if not expected:
        if cfg.drop_boundary_mode != 'root_sphere':
            raise ValueError('Policy ellipsoid does not match saved environment termination')
        return
    for name, value in expected.items():
        actual = getattr(cfg, name)
        if (tuple(actual) if isinstance(actual, (list, tuple)) else actual) != value:
            raise ValueError(f'Policy and environment drop boundary differ: {name}')


def verify_geometry_com(cfg, spec):
    if cfg.drop_boundary_mode == 'com_ellipsoid' and any(
            abs(a-b) > 1e-9 for a, b in zip(cfg.drop_object_com_local_m, spec['object_com'])):
        raise ValueError('Boundary COM differs from live object geometry')
