"""Stable policy interface for the vendored BrainCo Revo3 right hand.

The upstream model has 21 independent revolute joints with no coupling between them.
Mimic metadata on fixed decorative/tactile links does not add policy actions.
This ordering is shared by actions, position observations and velocity observations;
it deliberately does not depend on USD/PhysX traversal order.
"""

from pathlib import Path

REVO3_ASSET_DIR = Path(__file__).resolve().parent / "brainco" / "revo3_system"
REVO3_RUNTIME_USD = Path(__file__).resolve().with_name("revo3_right_instanceable.usda")
REVO3_MULTIPHYSICS_USD = Path(__file__).resolve().parent / "generated/revo3/revo3_right/revo3_right.usda"
REVO3_COMPACT_USD = (
    Path(__file__).resolve().parent
    / "generated/revo3_compact_visual_palm/revo3_right_compact_v4/revo3_right_compact_v4.usda"
)
REVO3_PALM_SDF_USD = Path(__file__).resolve().parent / "generated/revo3_palm_sdf/revo3_right_palm_sdf.usda"
REVO3_V14_CONVEX_USD = Path(__file__).resolve().parent / "generated/revo3_v14_touch_newton/revo3_v14_touch_convex/revo3_v14_touch_convex.usda"
REVO3_V14_TRAINING_USD = Path(__file__).resolve().parent / "generated/revo3_v14_training/revo3_v14_training.usda"
REVO3_JOINT_NAMES = (
    "right_thumb_CMP_joint",
    "right_thumb_CMR_joint",
    "right_thumb_MCP_joint",
    "right_thumb_PIP_joint",
    "right_thumb_DIP_joint",
    *(
        f"right_{finger}_{joint}_joint"
        for finger in ("index", "middle", "ring", "little")
        for joint in ("MPR", "MCP", "PIP", "DIP")
    ),
)
REVO3_DEFAULT_JOINT_POS = {
    "right_thumb_CMP_joint": 1.0,
    "right_thumb_CMR_joint": 0.5,
    "right_thumb_MCP_joint": 0.25,
    "right_thumb_PIP_joint": 0.25,
    "right_thumb_DIP_joint": 0.2,
    **{
        f"right_{finger}_{joint}_joint": position
        for finger in ("index", "middle", "ring", "little")
        for joint, position in (("MPR", 0.0), ("MCP", 0.25), ("PIP", 0.4), ("DIP", 0.25))
    },
}
