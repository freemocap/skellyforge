"""Standard human model — VRM-1.0-aligned humanoid skeleton.

The canonical human model defines every bone in the skeleton with its
T-pose reference geometry, coordinate frame, and twist resolution policy.
Aliases (for VRM/VMC wire names, Unreal bone names, etc.) live in
``human_bone_aliases.py`` — bones don't carry serialization knowledge.

Package structure:
    human_bones.py           — HumanBone, BoneReferenceGeometry,
                               CoordinateFrameDefinition, TwistPolicy
    human_bone_aliases.py    — BONE_ALIASES table + resolve_alias()
    human_blendshapes.py     — 52 ARKit blendshape channel declarations
    standard_human_model.py  — StandardHuman model (Pydantic, loaded once)
"""

from skellyforge.skellymodels.standard_human.human_bone_aliases import (
    BONE_ALIASES,
    resolve_alias,
    resolve_all_aliases,
)
from skellyforge.skellymodels.standard_human.human_blendshapes import (
    BlendShapeChannel,
    VRM_EXPRESSION_ARKIT_MAPPING,
    get_blendshape_count,
    get_blendshape_names,
)
from skellyforge.skellymodels.standard_human.human_bones import (
    BoneReferenceGeometry,
    CoordinateFrameDefinition,
    HumanBone,
    TwistPolicy,
    TwistTier,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    StandardHuman,
)

__all__ = [
    "HumanBone",
    "BoneReferenceGeometry",
    "CoordinateFrameDefinition",
    "TwistPolicy",
    "TwistTier",
    "BlendShapeChannel",
    "StandardHuman",
    "BONE_ALIASES",
    "resolve_alias",
    "resolve_all_aliases",
    "get_blendshape_count",
    "get_blendshape_names",
    "VRM_EXPRESSION_ARKIT_MAPPING",
]
