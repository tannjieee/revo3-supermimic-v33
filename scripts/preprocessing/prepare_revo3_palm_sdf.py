"""Prepare the 1200-face fixed-palm SDF from the official Revo3 conversion."""

import argparse
import json

from viserdex.assets.revo3_spec import REVO3_MULTIPHYSICS_USD
from viserdex.assets.prepare_revo3_palm_sdf import prepare_palm_sdf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default=str(REVO3_MULTIPHYSICS_USD))
parser.add_argument("--output_dir", default=str(REVO3_MULTIPHYSICS_USD.parents[2] / "revo3_palm_sdf"))
parser.add_argument("--faces", type=int, default=1200)
parser.add_argument("--sdf_resolution", type=int, default=512)
parser.add_argument("--voxel_size", type=float, default=0.0003)
parser.add_argument("--prepared_mesh", default=None)
args = parser.parse_args()
print(
    "PALM_SDF_ASSET_READY",
    json.dumps(
        prepare_palm_sdf(
            args.source,
            args.output_dir,
            face_count=args.faces,
            sdf_resolution=args.sdf_resolution,
            voxel_size=args.voxel_size,
            prepared_mesh=args.prepared_mesh,
        )
    ),
)
