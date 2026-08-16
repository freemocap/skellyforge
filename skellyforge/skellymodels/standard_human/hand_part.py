# skellyforge/skellymodels/standard_human/hand_part.py
"""The hand part: VRM 1.0 hand bones, authored once, instantiated per side.

16 segments over the 21 named hand keypoints. The ``wrist → *_mcp``
metacarpal/carpal span is not a VRM bone and is absorbed into ``hand`` (origin
``wrist``, exact target ``middle_finger_mcp``). VRM names the little finger
``little_*``; the keypoints name it ``pinky_*`` — the declaration is where
they meet.

Value provenance, per the plan's §7 honesty rules:

- ``rest_direction`` — the T-pose hand points +X with the fingers fanned in the
  horizontal plane, authored for the LEFT hand (the right side mirrors at
  reference-geometry build). Every hand segment declares its exact axis on y
  (+Y toward the child bone, the VRM humanoid rule), so the rest direction is
  the fanned finger direction: `_fan_direction(deg)` from +X toward −Y. Fan
  magnitudes from the Blender addon's ``freemocap_tpose`` (45/17/5.5/7.3/19
  degrees); signs from standard geometry — the authored-left thumb points
  toward the body midline (−Y). Refine if a sourced hand model appears.
- ``length_ratio`` — Buryanov & Kotiuk (2010) via the ``_BONE_LENGTH_RATIOS``
  table.
- ``rotation_limits`` None — Task 5 pins the local-frame convention (same
  rationale as the body part).
- Every hand segment carries a single exact axis (no approximate axis): they
  fall to the damped minimal-roll tier (the addon gives them no LockedTrack
  either).
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

# ── Rest directions: +X forward, fanned about Z (left hand), y-exact ─
def _fan_direction(degrees: float) -> tuple[float, float, float]:
    """The T-pose finger direction, fanned `degrees` from +X toward −Y (negative = toward +Y)."""
    theta = math.radians(degrees)
    return (math.cos(theta), -math.sin(theta), 0.0)

_REST_FORWARD = (1.0, 0.0, 0.0)
_FAN_THUMB = _fan_direction(45.0)
_FAN_INDEX = _fan_direction(17.0)
_FAN_RING = _fan_direction(-7.3)
_FAN_LITTLE = _fan_direction(-19.0)

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
            landmarks=(f"{keypoint_prefix}_mcp", f"{keypoint_prefix}_pip"),
            origin_landmark=f"{keypoint_prefix}_mcp",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, f"{keypoint_prefix}_pip", rest_direction=fan),
            ),
            length_ratio=_RATIO_FINGER_PROX,
        ),
        SegmentDefinition(
            name=f"{finger}_intermediate", parent=f"{finger}_proximal", parent_attachment=ParentAttachment.DISTAL,
            landmarks=(f"{keypoint_prefix}_pip", f"{keypoint_prefix}_dip"),
            origin_landmark=f"{keypoint_prefix}_pip",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, f"{keypoint_prefix}_dip", rest_direction=fan),
            ),
            length_ratio=_RATIO_FINGER_INT,
        ),
        SegmentDefinition(
            name=f"{finger}_distal", parent=f"{finger}_intermediate", parent_attachment=ParentAttachment.DISTAL,
            landmarks=(f"{keypoint_prefix}_dip", f"{keypoint_prefix}_tip"),
            origin_landmark=f"{keypoint_prefix}_dip",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, f"{keypoint_prefix}_tip", rest_direction=fan),
            ),
            length_ratio=_RATIO_FINGER_DIST,
        ),
    )


HAND_PART = SegmentPart(
    name="hand",
    segments=(
        SegmentDefinition(
            name="hand", parent="lower_arm", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("wrist", "middle_finger_mcp"), origin_landmark="wrist",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "middle_finger_mcp", rest_direction=_REST_FORWARD),
            ),
            length_ratio=_RATIO_HAND,
        ),
        SegmentDefinition(
            name="thumb_metacarpal", parent="hand", parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("thumb_cmc", "thumb_mcp"), origin_landmark="thumb_cmc",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "thumb_mcp", rest_direction=_FAN_THUMB),
            ),
            length_ratio=_RATIO_THUMB_MC,
        ),
        SegmentDefinition(
            name="thumb_proximal", parent="thumb_metacarpal", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("thumb_mcp", "thumb_ip"), origin_landmark="thumb_mcp",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "thumb_ip", rest_direction=_FAN_THUMB),
            ),
            length_ratio=_RATIO_THUMB_PROX,
        ),
        SegmentDefinition(
            name="thumb_distal", parent="thumb_proximal", parent_attachment=ParentAttachment.DISTAL,
            landmarks=("thumb_ip", "thumb_tip"), origin_landmark="thumb_ip",
            axes=(
                AxisDefinition("y", AxisKind.EXACT, "thumb_tip", rest_direction=_FAN_THUMB),
            ),
            length_ratio=_RATIO_THUMB_DIST,
        ),
        *_finger_segments("index", "index_finger", _FAN_INDEX),
        *_finger_segments("middle", "middle_finger", _REST_FORWARD),
        *_finger_segments("ring", "ring_finger", _FAN_RING),
        *_finger_segments("little", "pinky", _FAN_LITTLE),
    ),
)
