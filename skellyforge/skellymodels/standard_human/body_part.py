# skellyforge/skellymodels/standard_human/body_part.py
"""The body part: the VRM 1.0 torso and limb chains, authored side-agnostically.

Two parts compose the body:

- ``BODY_MIDLINE_PART`` — ``hips`` → ``head``, composed under the empty prefix.
- ``BODY_LIMB_PART`` — the shoulder/arm chain and the leg chain, authored once
  for the **left** side and instantiated ``left_``/``right_``. The right side's
  mirroring happens at reference-geometry build (negate Y, rebuild frames —
  SF-AL A3), never by reflecting a basis.

Cross-part references resolve by name agreement: ``shoulder`` attaches to the
midline ``upper_chest``, its twist references the midline ``neck_center``, and
``upper_leg`` branches from the midline ``hips``.

``hips`` and ``spine`` deliberately span the same endpoints
(``hips_center``→``trunk_center``) at the same ratio — the trunk piece
expressed at two VRM levels.

Value provenance, per the plan's §7 honesty rules:

- ``rest_rotation`` — derived from the canonical T-pose geometry (+Z up, +X
  forward, +Y = subject's left, arms out along ±Y, feet forward). Single-axis
  values, so any Euler convention agrees. Cross-checked against the Blender
  addon's ``freemocap_tpose`` ±90° side pattern; the addon's raw eulers are
  bone-space and not portable.
- ``rest_roll`` — 0.0 everywhere for now; the rest approximate axis is pinned
  when the reference geometry is built (Task 5). Twist is keypoint-driven.
- ``length_ratio`` — Winter (2009) / Drillis & Contini (1966) segment-length
  ratios of stature. Estimates state so.
- ``rotation_limits`` — ``None`` for all segments for now: the addon's LOCAL
  limits are expressed in its bone space and land once the segment local-frame
  convention is pinned (Task 5). Nothing enforces limits this workstream, so a
  wrong-frame number would be silent corruption.
"""

from __future__ import annotations

import math

from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    AxisKind,
    ParentAttachment,
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import (
    SegmentPart,
    compose_parts,
)

# ── Rest orientations (canonical T-pose geometry) ──────────────────
_REST_UP = (0.0, 0.0, 0.0)               # +Z (up) — the midline chain
_REST_LEFT = (-math.pi / 2, 0.0, 0.0)    # +Y (subject's left; authored side)
_REST_DOWN = (math.pi, 0.0, 0.0)         # −Z — the legs
_REST_FORWARD = (0.0, math.pi / 2, 0.0)  # +X — feet and toes

# ── Length ratios (of stature) ─────────────────────────────────────
# Winter (2009) + Drillis & Contini (1966) via _BONE_LENGTH_RATIOS.
_RATIO_HIPS = 0.145      # hips_to_spine — hips and spine span the same endpoints
_RATIO_CHEST = 0.100     # spine_to_chest
_RATIO_UPPER_CHEST = 0.055
_RATIO_NECK = 0.090      # upper_chest_to_neck (neck base → head center)
_RATIO_HEAD = 0.040      # neck_to_head (head center → top)
_RATIO_SHOULDER = 0.103  # ESTIMATED: half-biacromial 0.117 minus the SC offset's
                         # lateral component 0.06·W (W = biacromial width)
_RATIO_UPPER_ARM = 0.186
_RATIO_LOWER_ARM = 0.146
_RATIO_UPPER_LEG = 0.245
_RATIO_LOWER_LEG = 0.246
_RATIO_FOOT = 0.026     # ESTIMATED: 2:1 split of Winter's 0.039 ankle→toe at
_RATIO_TOES = 0.013     #           the metatarsophalangeal joint

# Axis declarations: the exact axis is always ``(origin, long_axis)`` — the
# segment's defining direction. The approximate axis is ``(origin, twist)`` —
# BOTH the reference geometry and the live solver derive the twist direction as
# origin → twist keypoint (see reference_geometry._rest_approximate_axis and
# orientation_solver), so the ``(from, to)`` pair must reproduce exactly that
# vector. The twist keypoint is a DIRECTION REFERENCE on every segment here — it
# stays OUT of ``rigid_points`` even where it is same-body (foot's heel),
# per the migration rule that keeps 2-point rigid sets everywhere this round.

BODY_MIDLINE_PART = SegmentPart(
    name="body_midline",
    segments=(
        SegmentDefinition(
            name="hips", parent=None, parent_attachment=ParentAttachment.ORIGIN,  # inert — root
            rigid_points=("hips_center", "trunk_center"), origin_keypoint="hips_center",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "hips_center", "trunk_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "hips_center", "right_hip"),
            ),
            rest_rotation=_REST_UP, rest_roll=0.0, length_ratio=_RATIO_HIPS,
        ),
        SegmentDefinition(
            name="spine", parent="hips", parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=("hips_center", "trunk_center"), origin_keypoint="hips_center",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "hips_center", "trunk_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "hips_center", "right_hip"),
            ),
            rest_rotation=_REST_UP, rest_roll=0.0, length_ratio=_RATIO_HIPS,
        ),
        SegmentDefinition(
            name="chest", parent="spine", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("trunk_center", "neck_center"), origin_keypoint="trunk_center",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "trunk_center", "neck_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "trunk_center", "right_shoulder"),
            ),
            rest_rotation=_REST_UP, rest_roll=0.0, length_ratio=_RATIO_CHEST,
        ),
        SegmentDefinition(
            name="upper_chest", parent="chest", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("mid_sternum", "neck_center"), origin_keypoint="mid_sternum",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "mid_sternum", "neck_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "mid_sternum", "right_shoulder"),
            ),
            rest_rotation=_REST_UP, rest_roll=0.0, length_ratio=_RATIO_UPPER_CHEST,
        ),
        SegmentDefinition(
            name="neck", parent="upper_chest", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("neck_center", "head_center"), origin_keypoint="neck_center",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "neck_center", "head_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "neck_center", "nose"),
            ),
            rest_rotation=_REST_UP, rest_roll=0.0, length_ratio=_RATIO_NECK,
        ),
        SegmentDefinition(
            name="head", parent="neck", parent_attachment=ParentAttachment.DISTAL,
            # The 7-point skull clique: this is what makes the head a FULL rigid
            # body (fixed pairwise distances), not a 2-point degenerate segment.
            # ``head_vertex`` is included deliberately — the model does not care
            # whether a point is mapping-derived today or tracker-measured
            # tomorrow; round-trip provenance is orthogonal to whether the point
            # is rigid with the skull. The jaw and mouth corners are NOT in the
            # set: they articulate.
            rigid_points=(
                "head_center", "head_vertex", "nose", "left_eye", "right_eye",
                "left_ear", "right_ear",
            ),
            origin_keypoint="head_center",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "head_center", "head_vertex"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "head_center", "nose"),
            ),
            rest_rotation=_REST_UP, rest_roll=0.0, length_ratio=_RATIO_HEAD,
        ),
    ),
)

BODY_LIMB_PART = SegmentPart(
    name="body_limb",
    segments=(
        SegmentDefinition(
            name="shoulder", parent="upper_chest", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("sternoclavicular", "shoulder"), origin_keypoint="sternoclavicular",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "sternoclavicular", "shoulder"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "sternoclavicular", "neck_center"),
            ),
            rest_rotation=_REST_LEFT, rest_roll=0.0, length_ratio=_RATIO_SHOULDER,
        ),
        SegmentDefinition(
            name="upper_arm", parent="shoulder", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("shoulder", "elbow"), origin_keypoint="shoulder",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),
                # the wrist is NOT rigid with the upper arm — the forearm rotates
                # about the elbow — so it is an external approximate-axis reference.
                AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),
            ),
            rest_rotation=_REST_LEFT, rest_roll=0.0, length_ratio=_RATIO_UPPER_ARM,
        ),
        SegmentDefinition(
            name="lower_arm", parent="upper_arm", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("elbow", "wrist"), origin_keypoint="elbow",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "elbow", "wrist"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "elbow", "middle_finger_mcp"),
            ),
            rest_rotation=_REST_LEFT, rest_roll=0.0, length_ratio=_RATIO_LOWER_ARM,
        ),
        SegmentDefinition(
            name="upper_leg", parent="hips", parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=("hip", "knee"), origin_keypoint="hip",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "hip", "knee"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "hip", "ankle"),
            ),
            rest_rotation=_REST_DOWN, rest_roll=0.0, length_ratio=_RATIO_UPPER_LEG,
        ),
        SegmentDefinition(
            name="lower_leg", parent="upper_leg", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("knee", "ankle"), origin_keypoint="knee",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "knee", "ankle"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "knee", "big_toe"),
            ),
            rest_rotation=_REST_DOWN, rest_roll=0.0, length_ratio=_RATIO_LOWER_LEG,
        ),
        SegmentDefinition(
            name="foot", parent="lower_leg", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("ankle", "foot_ball"), origin_keypoint="ankle",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "ankle", "foot_ball"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "ankle", "heel"),
            ),
            rest_rotation=_REST_FORWARD, rest_roll=0.0, length_ratio=_RATIO_FOOT,
        ),
        SegmentDefinition(
            name="toes", parent="foot", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("foot_ball", "big_toe"), origin_keypoint="foot_ball",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "foot_ball", "big_toe"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "foot_ball", "small_toe"),
            ),
            rest_rotation=_REST_FORWARD, rest_roll=0.0, length_ratio=_RATIO_TOES,
        ),
    ),
)


def compose_body_parts() -> list[SegmentDefinition]:
    """Compose the full body: the midline once, the limbs left and right."""
    return compose_parts(
        [
            (BODY_MIDLINE_PART, ""),
            (BODY_LIMB_PART, "left_"),
            (BODY_LIMB_PART, "right_"),
        ]
    )
