# skellyforge/skellymodels/standard_human/body_part.py
"""The body part: the VRM 1.0 torso and limb chains, authored side-agnostically.

Two parts compose the body:

- ``BODY_MIDLINE_PART`` — ``hips`` → ``head``, composed under the empty prefix.
- ``BODY_LIMB_PART`` — the shoulder/arm chain and the leg chain, authored once
  for the **left** side and instantiated ``left_``/``right_``. The right side's
  mirroring happens at reference-geometry build (negate Y, rebuild frames —
  SF-AL A3), never by reflecting a basis.

Cross-part references resolve by name agreement: ``shoulder`` attaches to the
midline ``upper_chest``, and ``upper_leg`` branches from the midline ``hips``.

``hips`` is a 4-point rigid body spanning ``hips_center``→``trunk_center`` with
the ``left_hip``/``right_hip`` pair; ``spine`` spans the same trunk endpoints
(``hips_center``→``trunk_center``) as a 2-point segment at the same ratio — the
trunk piece expressed at two VRM levels.

Authoring convention for the body (VRM 1.0 local frame, stated once here per
``segment_definition``'s header):

- Every body segment declares its EXACT axis on **y** — ``+Y`` points toward the
  child bone (the VRM 1.0 humanoid rule). The exact target is the
  toward-child endpoint (the segment's distal-facing endpoint).
- The APPROXIMATE axis (where one exists — hips, foot, toes) is declared on the
  axis whose Gram-Schmidt direction matches the segment's authored rest frame.
- The remaining axis follows the segment's own third point where one is
  declared, otherwise ``+X`` at the T-pose.

Value provenance, per the plan's §7 honesty rules:

- ``rest_direction`` — the world-space unit vector the declared **y** axis points
  along at the standard T-pose (+Z up, +X forward, +Y = subject's left, arms
  out along ±Y, feet forward, legs down): +Z up (midline), +Y left (arms), −Z
  down (legs), +X forward (feet/toes). Cross-checked against the Blender
  addon's ``freemocap_tpose`` side pattern; the addon's raw eulers are
  bone-space and not portable.
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

# ── Rest orientations (standard T-pose geometry, y-exact: R · ŷ = toward-child) ─
_REST_UP = (0.0, 0.0, 1.0)       # +Z (up) — the midline chain (+π/2 about X)
_REST_LEFT = (0.0, 1.0, 0.0)             # +Y (subject's left; authored side) — identity
_REST_DOWN = (0.0, 0.0, -1.0)    # −Z — the legs (−π/2 about X)
_REST_FORWARD = (1.0, 0.0, 0.0) # +X — feet and toes (−π/2 about Z)

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

# Axis declarations: the exact axis is always the segment's defining direction,
# resolved ``origin → target``, declared on y (+Y toward the child bone). An
# approximate axis is ``origin → twist``, where
# ``twist`` is a second point of the segment's own rigid geometry, declared on
# the axis whose Gram-Schmidt direction matches the segment's authored rest
# frame. The 3/4-point rigid segments (hips, foot, toes, head) carry both an
# exact and an approximate axis; the 2-point segments here carry only the exact
# axis, their roll resolved by the damped minimal-roll tier.

BODY_MIDLINE_PART = SegmentPart(
    name="body_midline",
    segments=(
        SegmentDefinition(
            name="hips", parent=None, parent_attachment=ParentAttachment.ORIGIN,  # inert — root
            # The hips are a FULL 4-point rigid body: the two trunk endpoints
            # (the exact direction) plus the two hip joints (the lateral pair
            # that resolves the pelvis roll).
            landmarks=("hips_center", "trunk_center", "left_hip", "right_hip"),
            origin_landmark="hips_center",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "trunk_center", rest_direction=_REST_UP),
                AxisDefinition("x", AxisKind.APPROXIMATE, "right_hip", rest_direction=(1.0, 0.0, 0.0)),
            ),
            length_ratio=_RATIO_HIPS,
        ),
        SegmentDefinition(
            name="spine", parent="hips", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("hips_center", "trunk_center"), origin_landmark="hips_center",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "trunk_center", rest_direction=_REST_UP),
            ),
            length_ratio=_RATIO_HIPS,
        ),
        SegmentDefinition(
            name="chest", parent="spine", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("trunk_center", "neck_center"), origin_landmark="trunk_center",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "neck_center", rest_direction=_REST_UP),
            ),
            length_ratio=_RATIO_CHEST,
        ),
        SegmentDefinition(
            name="upper_chest", parent="chest", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("mid_sternum", "neck_center"), origin_landmark="mid_sternum",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "neck_center", rest_direction=_REST_UP),
            ),
            length_ratio=_RATIO_UPPER_CHEST,
        ),
        SegmentDefinition(
            name="neck", parent="upper_chest", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("neck_center", "head_center"), origin_landmark="neck_center",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "head_center", rest_direction=_REST_UP),
            ),
            length_ratio=_RATIO_NECK,
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
            landmarks=(
                "head_center", "head_vertex", "nose", "left_eye", "right_eye",
                "left_ear", "right_ear",
            ),
            origin_landmark="head_center",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "head_vertex", rest_direction=_REST_UP),
                AxisDefinition("z", AxisKind.APPROXIMATE, "nose", rest_direction=(1.0, 0.0, 0.0)),
            ),
            length_ratio=_RATIO_HEAD,
        ),
    ),
)

BODY_LIMB_PART = SegmentPart(
    name="body_limb",
    segments=(
        SegmentDefinition(
            name="shoulder", parent="upper_chest", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("sternoclavicular", "shoulder"), origin_landmark="sternoclavicular",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "shoulder", rest_direction=_REST_LEFT),
            ),
            length_ratio=_RATIO_SHOULDER,
        ),
        SegmentDefinition(
            name="upper_arm", parent="shoulder", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("shoulder", "elbow"), origin_landmark="shoulder",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "elbow", rest_direction=_REST_LEFT),
            ),
            length_ratio=_RATIO_UPPER_ARM,
        ),
        SegmentDefinition(
            name="lower_arm", parent="upper_arm", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("elbow", "wrist"), origin_landmark="elbow",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "wrist", rest_direction=_REST_LEFT),
            ),
            length_ratio=_RATIO_LOWER_ARM,
        ),
        SegmentDefinition(
            name="upper_leg", parent="hips", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("hip", "knee"), origin_landmark="hip",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "knee", rest_direction=_REST_DOWN),
            ),
            length_ratio=_RATIO_UPPER_LEG,
        ),
        SegmentDefinition(
            name="lower_leg", parent="upper_leg", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("knee", "ankle"), origin_landmark="knee",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "ankle", rest_direction=_REST_DOWN),
            ),
            length_ratio=_RATIO_LOWER_LEG,
        ),
        SegmentDefinition(
            name="foot", parent="lower_leg", parent_attachment=ParentAttachment.DISTAL,
            # The foot is a FULL 3-point rigid body: the exact axis (ankle →
            # foot_ball) plus the heel (the posterior point that resolves the
            # foot's roll-and-pitch reference). The heel points down-back, so
            # the foot's approximate axis lands on z (down after Gram-Schmidt).
            landmarks=("ankle", "foot_ball", "heel"), origin_landmark="ankle",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "foot_ball", rest_direction=_REST_FORWARD),
                AxisDefinition("z", AxisKind.APPROXIMATE, "heel", rest_direction=(0.0, 0.0, -1.0)),
            ),
            length_ratio=_RATIO_FOOT,
        ),
        SegmentDefinition(
            name="toes", parent="foot", parent_attachment=ParentAttachment.DISTAL,
            # The toes are a FULL 3-point rigid body: the exact axis (foot_ball →
            # big_toe) plus small_toe (the lateral point that resolves the toes'
            # roll reference). The small_toe points laterally, so the toes'
            # approximate axis lands on x (lateral after Gram-Schmidt).
            landmarks=("foot_ball", "big_toe", "small_toe"),
            origin_landmark="foot_ball",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "big_toe", rest_direction=_REST_FORWARD),
                AxisDefinition("x", AxisKind.APPROXIMATE, "small_toe", rest_direction=(0.0, 1.0, 0.0)),
            ),
            length_ratio=_RATIO_TOES,
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
