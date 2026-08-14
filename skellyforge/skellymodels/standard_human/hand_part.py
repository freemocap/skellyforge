# skellyforge/skellymodels/standard_human/hand_part.py
"""The hand part: VRM 1.0 hand bones, authored once, instantiated per side.

16 segments over the 21 named hand keypoints. The ``wrist → *_mcp``
metacarpal/carpal span is not a VRM bone and is absorbed into ``hand`` (origin
``wrist``, long axis ``middle_finger_mcp``). VRM names the little finger
``little_*``; the keypoints name it ``pinky_*`` — the declaration is where
they meet.

Value provenance, per the plan's §7 honesty rules:

- ``rest_rotation`` — the T-pose hand points +X with the fingers fanned in the
  horizontal plane, authored for the LEFT hand (the right side mirrors at
  reference-geometry build). Sourced: fan magnitudes from the Blender addon's
  ``freemocap_tpose`` (45/17/5.5/7.3/19 degrees); signs from canonical geometry
  — the authored-left thumb points toward the body midline (−Y). Refine if a
  sourced hand model appears.
- ``length_ratio`` — Buryanov & Kotiuk (2010) via the ``_BONE_LENGTH_RATIOS``
  table.
- ``rest_roll`` 0.0 and ``rotation_limits`` None — Task 5 pins the local-frame
  convention (same rationale as the body part).
- Fingers carry a single exact axis (no approximate axis): they fall to the
  damped minimal-roll tier (the addon gives them no LockedTrack either).
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

# ── Rest orientations: +X forward, fanned about Z (left hand) ──────
_REST_FORWARD = (0.0, math.pi / 2, 0.0)  # +X — the hand and the middle finger
_FAN_THUMB = (0.0, math.pi / 2, -math.radians(45.0))
_FAN_INDEX = (0.0, math.pi / 2, -math.radians(17.0))
_FAN_RING = (0.0, math.pi / 2, math.radians(7.3))
_FAN_LITTLE = (0.0, math.pi / 2, math.radians(19.0))

# ── Length ratios (of stature) ─────────────────────────────────────
# Buryanov & Kotiuk (2010).
_RATIO_HAND = 0.050        # wrist → MCP span (finger_metacarpal)
_RATIO_THUMB_MC = 0.015
_RATIO_THUMB_PROX = 0.018
_RATIO_THUMB_DIST = 0.017
_RATIO_FINGER_PROX = 0.028
_RATIO_FINGER_INT = 0.018
_RATIO_FINGER_DIST = 0.014


def _finger_segments(
    finger: str,
    keypoint_prefix: str,
    fan: tuple[float, float, float],
) -> tuple[SegmentDefinition, SegmentDefinition, SegmentDefinition]:
    """The three VRM phalanges of one finger, keypoints per the tracker names."""
    return (
        SegmentDefinition(
            name=f"{finger}_proximal", parent="hand", parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=(f"{keypoint_prefix}_mcp", f"{keypoint_prefix}_pip"),
            origin_keypoint=f"{keypoint_prefix}_mcp",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, f"{keypoint_prefix}_mcp", f"{keypoint_prefix}_pip"),
            ),
            rest_rotation=fan, rest_roll=0.0, length_ratio=_RATIO_FINGER_PROX,
        ),
        SegmentDefinition(
            name=f"{finger}_intermediate", parent=f"{finger}_proximal", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=(f"{keypoint_prefix}_pip", f"{keypoint_prefix}_dip"),
            origin_keypoint=f"{keypoint_prefix}_pip",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, f"{keypoint_prefix}_pip", f"{keypoint_prefix}_dip"),
            ),
            rest_rotation=fan, rest_roll=0.0, length_ratio=_RATIO_FINGER_INT,
        ),
        SegmentDefinition(
            name=f"{finger}_distal", parent=f"{finger}_intermediate", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=(f"{keypoint_prefix}_dip", f"{keypoint_prefix}_tip"),
            origin_keypoint=f"{keypoint_prefix}_dip",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, f"{keypoint_prefix}_dip", f"{keypoint_prefix}_tip"),
            ),
            rest_rotation=fan, rest_roll=0.0, length_ratio=_RATIO_FINGER_DIST,
        ),
    )


HAND_PART = SegmentPart(
    name="hand",
    segments=(
        SegmentDefinition(
            name="hand", parent="lower_arm", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("wrist", "middle_finger_mcp"), origin_keypoint="wrist",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "wrist", "middle_finger_mcp"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "wrist", "thumb_cmc"),
            ),
            rest_rotation=_REST_FORWARD, rest_roll=0.0, length_ratio=_RATIO_HAND,
        ),
        SegmentDefinition(
            name="thumb_metacarpal", parent="hand", parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=("thumb_cmc", "thumb_mcp"), origin_keypoint="thumb_cmc",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "thumb_cmc", "thumb_mcp"),
            ),
            rest_rotation=_FAN_THUMB, rest_roll=0.0, length_ratio=_RATIO_THUMB_MC,
        ),
        SegmentDefinition(
            name="thumb_proximal", parent="thumb_metacarpal", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("thumb_mcp", "thumb_ip"), origin_keypoint="thumb_mcp",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "thumb_mcp", "thumb_ip"),
            ),
            rest_rotation=_FAN_THUMB, rest_roll=0.0, length_ratio=_RATIO_THUMB_PROX,
        ),
        SegmentDefinition(
            name="thumb_distal", parent="thumb_proximal", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("thumb_ip", "thumb_tip"), origin_keypoint="thumb_ip",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "thumb_ip", "thumb_tip"),
            ),
            rest_rotation=_FAN_THUMB, rest_roll=0.0, length_ratio=_RATIO_THUMB_DIST,
        ),
        *_finger_segments("index", "index_finger", _FAN_INDEX),
        *_finger_segments("middle", "middle_finger", _REST_FORWARD),
        *_finger_segments("ring", "ring_finger", _FAN_RING),
        *_finger_segments("little", "pinky", _FAN_LITTLE),
    ),
)
