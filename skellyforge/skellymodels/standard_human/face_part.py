# skellyforge/skellymodels/standard_human/face_part.py
"""The face part: driven VRM 1.0 face bones + FreeMoCap face-detail segments
+ declared-but-null blendshapes.

The face part holds 8 segments — 3 driven VRM 1.0 face bones plus 5 FreeMoCap
face-detail segments (nose, ears, mouth corners) — plus the 52 declared-null
blendshape channels.

``left_eye``, ``right_eye`` and ``jaw`` are DRIVEN segments (VRM 1.0
humanoid.md: "the model's eye movement controlled by bones" — eyes and jaw are
defined bones parented to ``head``).

``nose``, ``left_ear``, ``right_ear``, ``left_mouth`` and ``right_mouth`` are
FreeMoCap face-detail segments, beyond the VRM humanoid set. They exist because
the face's tracked keypoints give real head-axis references (the anterior
``nose``, the lateral ``left_ear``/``right_ear``, and the mouth corners) that
VRM 1.0's humanoid has no bones for. They branch from the head's ORIGIN (the
head-center line) and each enters ``required_landmarks()`` like any other
segment.

The eyes / ears / nose are **rigid children** of the head: all their landmarks
are skull-clique members, so they inherit the head's solved pose (declared
``rigid_with_parent``) instead of solving independently from two noisy face
points. The **jaw** and the **mouth corners** articulate (they anchor at
observed) and keep their independent solves.

The 52 ARKit blendshape channels compose alongside the skeleton (SF-AL A4),
stay declared-but-null (locked decision 4 holds for the channels), and come
from ``skellyforge.skellymodels.standard_human.human_blendshapes``.

Authoring convention for the face (VRM 1.0 local frame, per
``segment_definition``'s header):

- The driven VRM face bones (``left_eye``/``right_eye``/``jaw``) declare their
  exact axis on **z** — ``+Z`` is the gaze direction, target ``nose``.
- The face-detail segments share the head's VRM frame (+Y up, +Z face-forward,
  +X right): ``nose``/``left_mouth``/``right_mouth`` declare exact on **z**
  (forward-facing), and ``left_ear``/``right_ear`` declare exact on **x**
  (lateral).
- Each ``rest_rotation`` extrudes the declared axis name through its euler so
  the rest frame's named vector is the authored gaze/lateral direction.
"""

from __future__ import annotations

import math

from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    AxisKind,
    ParentAttachment,
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import SegmentPart

# Face bones + face detail declare their exact axis on z (gaze, +Z), so the
# euler must extrude +Z toward the authored direction. Gaze is anterior (+X) at
# the T-pose → ``R·ẑ = +X`` via a +90° Y euler. The mouth/jaw carry small
# offsets (their corner/chin directions), encoded in the same +Z-extruded euler.
_EYE_REST = (0.0, math.pi / 2, 0.0)                    # gaze: +Z → +X (anterior)
_NOSE_REST = (0.0, math.pi / 2, 0.0)                   # forward: +Z → +X

# The jaw and mouth-corner rest directions are AUTHORED here, from the
# facial-proportions canon (eye width as the facial-third unit), independent of
# any tracker mapping:
#   · nose → jaw        ≈ 0.9·eye-width down, 0.2·eye-width posterior;
#   · nose → mouth corner ≈ 0.35·eye-width down, 0.2·eye-width posterior,
#                           ±0.3·eye-width lateral (left/right).
# ESTIMATED, not published — refine if a sourced value appears. The tracker
# mappings derive the same named points (jaw, mouth corners) from the same
# canon on their side of the boundary; the two runtime sides are independent by
# design, and their numeric agreement is pinned by the consistency test
# (test_face_mapping_consistency.py), not by a shared constant.
# jaw→nose: (0.2, 0, 0.9)/√(0.2²+0.9²) anterior/down; the restoring angle from
# +Z (toward +X) is asin(0.2 / sqrt(0.2² + 0.9²)).
_JAW_REST = (0.0, math.asin(0.2 / math.sqrt(0.2**2 + 0.9**2)), 0.0)

# corner→nose unit direction: (0.2, ∓0.3, 0.35)/√(0.2²+0.3²+0.35²) — anterior
# 0.2 / lateral 0.3 / down 0.35 × eye_width, the facial-proportions canon
# estimate (same provenance as the jaw).
_MOUTH_BETA = math.asin(0.2 / math.sqrt(0.2**2 + 0.3**2 + 0.35**2))
_MOUTH_ALPHA = math.asin(0.3 / (math.sqrt(0.2**2 + 0.3**2 + 0.35**2) * math.cos(_MOUTH_BETA)))
# left mouth corner sits at +Y, so nose − left_mouth has −Y: −cos β · sin α < 0 ⇒ α > 0.
# Authored for the LEFT side; the right_mouth shares this rest_rotation and the
# reference geometry mirrors it to +Y, exactly like the body/hand limbs.
_MOUTH_REST = (+_MOUTH_ALPHA, +_MOUTH_BETA, 0.0)

# ears: exact axis on x (lateral). Authored for the LEFT ear (+Y, subject's
# left): ``R·x̂ = +Y`` via a +90° Z euler. The right ear shares this
# rest_rotation and the reference geometry mirrors it to −Y (SF-AL A3).
_EAR_REST = (0.0, 0.0, math.pi / 2)

# nominal — face segments branch from the head origin (zero-length branch);
# the value only satisfies the > 0 validation
_RATIO_NOMINAL = 0.01

FACE_PART = SegmentPart(
    name="face",
    segments=(
        SegmentDefinition(
            name="left_eye", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("left_eye", "nose"), origin_landmark="left_eye",
            axes=(AxisDefinition("z", AxisKind.EXACT, "nose"),),
            rest_rotation=_EYE_REST, length_ratio=_RATIO_NOMINAL,
            rigid_with_parent=True,
        ),
        SegmentDefinition(
            name="right_eye", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("right_eye", "nose"), origin_landmark="right_eye",
            axes=(AxisDefinition("z", AxisKind.EXACT, "nose"),),
            rest_rotation=_EYE_REST, length_ratio=_RATIO_NOMINAL,
            rigid_with_parent=True,
        ),
        SegmentDefinition(
            name="jaw", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("jaw", "nose"), origin_landmark="jaw",
            axes=(AxisDefinition("z", AxisKind.EXACT, "nose"),),
            rest_rotation=_JAW_REST, length_ratio=_RATIO_NOMINAL,
        ),
        SegmentDefinition(
            name="nose", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("head_center", "nose"), origin_landmark="head_center",
            axes=(AxisDefinition("z", AxisKind.EXACT, "nose"),),
            rest_rotation=_NOSE_REST, length_ratio=_RATIO_NOMINAL,
            rigid_with_parent=True,
        ),
        SegmentDefinition(
            name="left_ear", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("head_center", "left_ear"), origin_landmark="head_center",
            axes=(AxisDefinition("x", AxisKind.EXACT, "left_ear"),),
            rest_rotation=_EAR_REST, length_ratio=_RATIO_NOMINAL,
            rigid_with_parent=True,
        ),
        SegmentDefinition(
            name="right_ear", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("head_center", "right_ear"), origin_landmark="head_center",
            axes=(AxisDefinition("x", AxisKind.EXACT, "right_ear"),),
            rest_rotation=_EAR_REST, length_ratio=_RATIO_NOMINAL,
            rigid_with_parent=True,
        ),
        SegmentDefinition(
            name="left_mouth", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("left_mouth", "nose"), origin_landmark="left_mouth",
            axes=(AxisDefinition("z", AxisKind.EXACT, "nose"),),
            rest_rotation=_MOUTH_REST, length_ratio=_RATIO_NOMINAL,
        ),
        SegmentDefinition(
            name="right_mouth", parent="head", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("right_mouth", "nose"), origin_landmark="right_mouth",
            axes=(AxisDefinition("z", AxisKind.EXACT, "nose"),),
            rest_rotation=_MOUTH_REST, length_ratio=_RATIO_NOMINAL,
        ),
    ),
)
