"""Standard human model — VRM-1.0-aligned humanoid skeleton.

The standard human model is the composed 60-segment human: segments authored
once and expanded into a flat indexed list, with the T-pose reference geometry
built against it. Aliases (for VRM/VMC wire names, Unreal bone names, etc.)
live in ``human_bone_aliases.py`` — segments don't carry serialization
knowledge.

A **landmark** is a named point in a segment's local frame: it has a static
rest definition (a T-pose position in the reference geometry) and a per-frame
world hydration (the mapping hydrates its name from tracker keypoints).

Package structure:
    segment_definition.py   — SegmentDefinition (frozen; origin/long/twist)
    segment_parts.py        — SegmentPart, compose_parts
    body_part.py            — BODY_MIDLINE_PART, BODY_LIMB_PART
    hand_part.py            — HAND_PART
    face_part.py            — FACE_PART (3 driven VRM 1.0 face bones + 5 FreeMoCap face-detail segments)
    reference_geometry.py   — ReferenceGeometry, SegmentReferenceGeometry
    human_bone_aliases.py   — BONE_ALIASES table + resolve_alias()
    human_blendshapes.py    — 52 ARKit blendshape channel declarations
    standard_human_model.py — StandardHuman (composed) + compose_standard_human
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
from skellyforge.skellymodels.standard_human.reference_geometry import (
    ReferenceGeometry,
    SegmentReferenceGeometry,
    build_reference_geometry,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    StandardHuman,
    compose_standard_human,
)
from skellyforge.skellymodels.standard_human.segment_definition import (
    ParentAttachment,
    RotationLimits,
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import (
    SegmentPart,
    compose_parts,
)

__all__ = [
    "SegmentDefinition",
    "SegmentPart",
    "ParentAttachment",
    "RotationLimits",
    "compose_parts",
    "StandardHuman",
    "compose_standard_human",
    "ReferenceGeometry",
    "SegmentReferenceGeometry",
    "build_reference_geometry",
    "BlendShapeChannel",
    "BONE_ALIASES",
    "resolve_alias",
    "resolve_all_aliases",
    "get_blendshape_count",
    "get_blendshape_names",
    "VRM_EXPRESSION_ARKIT_MAPPING",
]
