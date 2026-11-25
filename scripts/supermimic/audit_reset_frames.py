"""Offline world-frame reset regression; never steps physics or changes assets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/viserdex'))
from viserdex.assets.prepare_manual_convex import MANUAL_USD
from viserdex.supermimic.geometry_asset import build_geometry_spec
from viserdex.supermimic.object_variant import VARIANTS
from viserdex.supermimic.support_placement import SupportPlacement
from viserdex.supermimic.wuji_pose import HOME_POSITION
from viserdex.distillation.noise import quat_apply, quat_conjugate, quat_mul


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    robot = MANUAL_USD.resolve()
    query = SupportPlacement(robot, build_geometry_spec(robot, VARIANTS['die50']))
    rng = np.random.default_rng(args.seed)
    base_quat = np.array([0., -2**-.5, 0., 2**-.5], dtype=np.float32)
    inverse = quat_conjugate(torch.tensor(base_quat, dtype=torch.float32))
    rows = []
    for i in range(args.samples):
        base_pos = np.array([.12+2*(i % 32), 2*((i//32) % 32), .51], dtype=np.float32)
        q = (np.asarray(HOME_POSITION)+rng.uniform(-.03, .03, 21)).astype(np.float32)
        position = base_pos+np.array([-.1, -.01, .055])+rng.uniform([-.02,-.02,0],[.02,.02,0])
        quat = Rotation.from_euler('xyz', rng.uniform(-3.14,3.14,3)).as_quat()
        # Runtime reset buffers are already float32 before placement. Preserve
        # their actual lateral coordinates instead of rounding them only after.
        pose, audit = query.place_world(q, np.r_[position,quat].astype(np.float32), base_pos, base_quat, [0,0,1])
        actual = torch.tensor(pose, dtype=torch.float32)
        local_p = quat_apply(inverse, actual[:3]-torch.tensor(base_pos,dtype=torch.float32))
        local_q = quat_mul(inverse, actual[3:])
        query.set_pose(q, local_p.numpy(), local_q.numpy())
        gap = query.minimum_gap()
        rows.append(dict(sample=i, cast_gap_m=audit['gap_m'], written_gap_m=gap))
    gaps = np.array([r['written_gap_m'] for r in rows])
    result = dict(protocol='world_xyzw_v2', samples=args.samples, seed=args.seed,
                  min_gap_m=float(gaps.min()), max_gap_m=float(gaps.max()),
                  passed=bool(np.isfinite(gaps).all() and (gaps>=-5e-6).all() and (gaps<=1e-5).all()),
                  rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
