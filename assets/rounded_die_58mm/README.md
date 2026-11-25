# 58 mm Newton training die

Generated from `/home/brainco/reorientation/assets/dice_58mm_isaaclab/dice_58mm.usda`
using `scripts/preprocessing/prepare_die_58mm.py`. All source points, faces,
materials, center of mass, principal moments and axes are retained. The only
physics schema addition is the existing Newton 200-vertex hull budget.
Source assets are untouched. Load at SI scale 1; do not scale again.

Nominal mass is **73.362112 g, estimated**, carried from the supplied package:
47 g at 50 mm times `(58/50)^3`. It is not a measured 58 mm mass. Existing
training mass/friction/gain randomization remains enabled.

New manual-convex runs default to `die58`; `--object_variant die58` selects it
explicitly. Saved environments retain their own object, and cross-object
checkpoint resume is rejected. Actor transfer, if later desired, is a distinct
experiment because both geometry and dynamics changed.

The fixed-palm reset has a separate 58 mm pose, derived offline against the
actual `generated/revo3_manual_convex/revo3_right.usda` collision meshes. HOME
joint angles are unchanged. Live reset checks require at least two palm hulls
and four total hand hull contacts, no penetration, and exact pose readback.
The old 50 mm fixed pose and its three-palm-support requirement remain intact.

Validation: 56 regression tests; 16 environments × 550 steps HOME holding with
randomized physical parameters, zero drops, maximum position drift 0.265 mm,
orientation drift 0.344 degrees, selected/unselected reset isolation checked.
An 8-environment v33 two-update preflight passed with real contact labels,
independent failure optimizer and CUDA Graph execution. These checks validate
the training path, not learned 58 mm task performance.
