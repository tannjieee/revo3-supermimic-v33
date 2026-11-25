"""Explicit object selection for new training; saved environments keep their object."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VARIANTS = {'die50': ROOT/'assets/rounded_die_50mm/object_newton.usda',
            'die50_47g': ROOT/'assets/rounded_die_50mm_47g/object_newton.usda',
            'die58': ROOT/'assets/rounded_die_58mm/object_newton.usda',
            'die70': ROOT/'assets/rounded_die/object_newton.usda'}


def bind_object(cfg, requested=None, *, saved=False):
    if saved:
        if requested is not None and Path(cfg.scene.object.spawn.usd_path).resolve() != VARIANTS[requested].resolve():
            raise ValueError('Saved object differs from --object_variant; use a fresh run for a different die')
        return
    path = VARIANTS[requested or 'die58'].resolve(strict=True)
    cfg.scene.object.spawn.usd_path = str(path)
    cfg.scene.object.spawn.scale = None  # SI dimensions and inertia already baked into USD.
    goal = cfg.commands.object_pose.goal_pose_visualizer_cfg.markers['goal']
    goal.usd_path = str(path)
    goal.scale = (1., 1., 1.)


def object_metadata(path):
    path = Path(path).resolve(strict=True)
    result = dict(usd_path=str(path), usd_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result['variant'] = next((name for name,p in VARIANTS.items() if p.resolve() == path), 'saved_custom')
    if result['variant'] in ('die50', 'die50_47g', 'die58'):
        manifest = json.loads(path.with_name('manifest.json').read_text())
        if result['usd_sha256'] != manifest['usd_sha256']:
            raise ValueError('Die asset differs from its generated manifest')
        result.update(dimensions_m=manifest['dimensions_m'], mass_kg=manifest['mass_kg'],
                      mass_source=manifest['mass_source'], material_srgb=manifest['material_srgb'])
        if 'mass_is_estimated' in manifest:
            result['mass_is_estimated'] = manifest['mass_is_estimated']
    return result
