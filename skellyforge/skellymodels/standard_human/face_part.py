# skellyforge/skellymodels/standard_human/face_part.py
"""The face part: driven VRM 1.0 face bones + declared-but-null blendshapes.

``left_eye``, ``right_eye`` and ``jaw`` are DRIVEN segments (VRM 1.0
humanoid.md: "the model's eye movement controlled by bones" — eyes and jaw are
defined bones parented to ``head``). They branch from the head's ORIGIN (the
head-center line) and each enters ``required_keypoints()`` like any other
segment. The 52 ARKit blendshape channels compose alongside the skeleton
(SF-AL A4), stay declared-but-null (locked decision 4 holds for the channels),
and come from
``skellyforge.skellymodels.standard_human.human_blendshapes``.
"""

from __future__ import annotations

import math

from skellyforge.skellymodels.standard_human.segment_definition import (
    ParentAttachment,
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import SegmentPart

# eyes: eye→nose axis is anterior (+X) at the T-pose → extrude the +Z rest axis
# to +X via a +90° Y euler
_EYE_REST = (0.0, math.pi / 2, 0.0)
# jaw: derived from the jaw-offset design — nose→jaw ≈ 0.9·eye-width down,
# 0.2·eye-width posterior, so jaw→nose ≈ (0.217, 0, 0.976); the restoring angle
# from +Z (toward +X) is asin(0.2 / sqrt(0.2² + 0.9²))
_JAW_REST = (0.0, math.asin(0.2 / math.sqrt(0.2**2 + 0.9**2)), 0.0)
# nominal — eyes and jaw branch from the head origin (zero-length branch);
# the value only satisfies the > 0 validation
_RATIO_NOMINAL = 0.01

FACE_PART = SegmentPart(
    name="face",
    segments=(
        SegmentDefinition(
            name="left_eye", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            origin_keypoint="left_eye", long_axis_keypoint="nose", twist_keypoint=None,
            rest_rotation=_EYE_REST, rest_roll=0.0, length_ratio=_RATIO_NOMINAL,
        ),
        SegmentDefinition(
            name="right_eye", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            origin_keypoint="right_eye", long_axis_keypoint="nose", twist_keypoint=None,
            rest_rotation=_EYE_REST, rest_roll=0.0, length_ratio=_RATIO_NOMINAL,
        ),
        SegmentDefinition(
            name="jaw", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            origin_keypoint="jaw", long_axis_keypoint="nose", twist_keypoint=None,
            rest_rotation=_JAW_REST, rest_roll=0.0, length_ratio=_RATIO_NOMINAL,
        ),
    ),
)
