"""Standard human model — VRM-1.0-aligned humanoid skeleton.

The current architecture is the seven-layer ontology (keypoint → mapping →
landmark → segment → linkage → chain → skeleton), defined in YAML and compiled
into typed objects whose references are objects, not strings. The composed
standard human is **95 segments / 94 linkages / 25 chains / 146 landmarks**;
its T-pose reference geometry is built by ``kinematics.build_standard_human_tpose``.

A **landmark** is a named point in a segment's local frame: it has a static
rest definition (a T-pose position) and a per-frame world hydration (the mapping
hydrates its name from tracker keypoints).

Current package structure:
    anatomical_landmark.py    — AnatomicalLandmark
    rigid_body_segment.py     — RigidBodySegment (+ AxisDefinition)
    joint_linkage.py          — JointLinkage (derived from parent edges)
    kinematic_chain.py        — KinematicChain (start → end path)
    human_skeleton.py         — HumanSkeleton.from_yaml (parts + sidedness + $include)
    standard_human_tpose.py   — StandardHumanTPose / SegmentTposeGeometry
    face_blendshapes.py       — FaceBlendShapes (52 ARKit blendshapes)
    definitions/              — YAML parts: standard_human, pelvis, axial, arm, hand, leg, foot, face

Old-architecture modules (``segment_definition.py`` / ``reference_geometry.py`` /
``standard_human_model.py`` / ``*_part.py`` / ``human_bone_aliases.py`` /
``human_blendshapes.py``) are still imported below for the not-yet-migrated
charuco + posthoc consumers, and are deleted once those migrate.
"""

from skellyforge.skellymodels.standard_human.anatomical_landmark import (
    AnatomicalLandmark,
)
from skellyforge.skellymodels.standard_human.rigid_body_segment import (
    RigidBodySegment,
)
from skellyforge.skellymodels.standard_human.axis_definition import AxisDefinition
from skellyforge.skellymodels.standard_human.joint_linkage import JointLinkage
from skellyforge.skellymodels.standard_human.kinematic_chain import KinematicChain
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton
from skellyforge.skellymodels.standard_human.face_blendshapes import FaceBlendShapes

# NOTE: StandardHumanTPose lives in standard_human_tpose.py but is intentionally
# NOT re-exported here — it imports kinematics (for RotationQuaternion), which
# imports the solver, which imports it back. Import it from kinematics or the
# module directly to avoid that cycle.

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
from skellyforge.skellymodels.standard_human.dead_reference_geometry import (
    ReferenceGeometry,
    SegmentReferenceGeometry,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    StandardHuman,
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
    # current architecture (the seven-layer ontology)
    "AnatomicalLandmark",
    "AxisDefinition",
    "RigidBodySegment",
    "JointLinkage",
    "KinematicChain",
    "HumanSkeleton",
    "FaceBlendShapes",
    # old architecture (being retired with the charuco + posthoc migration)
    "SegmentDefinition",
    "SegmentPart",
    "ParentAttachment",
    "RotationLimits",
    "compose_parts",
    "StandardHuman",
    "ReferenceGeometry",
    "SegmentReferenceGeometry",
    "BlendShapeChannel",
    "get_blendshape_count",
    "get_blendshape_names",
    "VRM_EXPRESSION_ARKIT_MAPPING",
]
