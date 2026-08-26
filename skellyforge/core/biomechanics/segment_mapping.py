"""Map the skeleton's segments onto de Leva's 16 anatomical segments.

The de Leva table divides the body into 16 anatomical segments (head+neck, three trunk
parts, and six bilateral limb parts). The skeleton models the same body as more rigid
segments than that - several of which make up one anatomical segment (a hand is a carpal cluster
plus nineteen phalanges/metacarpals; the head+neck is the skull plus the cervical spine).
This module states that mapping and, from it, distributes each anatomical segment's mass
across its skeleton segments proportionally to length cubed - the equal-density
approximation, which is the right default for the hand's many small bones.
"""

from __future__ import annotations

from collections.abc import Mapping

from skellyforge.core.biomechanics.anthropometric_parameters import AnthropometricParameters
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


_ANATOMICAL_BY_SKELETON_SEGMENT: dict[str, str] = {
    "skull": "head_neck",
    "cervical_spine": "head_neck",
    "thoracic": "upper_trunk",
    "left_clavicle": "upper_trunk",
    "right_clavicle": "upper_trunk",
    "sacrolumbar": "middle_trunk",
    "pelvis": "lower_trunk",
    "left_upper_arm": "upper_arm",
    "right_upper_arm": "upper_arm",
    "left_lower_arm": "forearm",
    "right_lower_arm": "forearm",
    "left_carpals": "hand",
    "right_carpals": "hand",
    "left_upper_leg": "thigh",
    "right_upper_leg": "thigh",
    "left_lower_leg": "shank",
    "right_lower_leg": "shank",
    "left_foot": "foot",
    "right_foot": "foot",
    "left_heel": "foot",
    "right_heel": "foot",
    "left_toes": "foot",
    "right_toes": "foot",
}


def anatomical_segment_name(*, skeleton_segment_name: str) -> str:
    """The de Leva anatomical segment a skeleton segment belongs to.

    The hand's phalanx and metacarpal segments are matched by name rather than listed
    exhaustively (there are nineteen per side), because every one of them is a hand bone.
    """
    mapped = _ANATOMICAL_BY_SKELETON_SEGMENT.get(skeleton_segment_name)
    if mapped is not None:
        return mapped
    if "phalanx" in skeleton_segment_name or "metacarpal" in skeleton_segment_name:
        return "hand"
    raise KeyError(
        f"unknown skeleton segment {skeleton_segment_name!r} - cannot map it to an "
        f"anatomical segment"
    )


def map_skeleton_segments(*, skeleton: SkeletonDefinition) -> dict[str, str]:
    """Every skeleton segment mapped to its de Leva anatomical segment name."""
    return {
        name: anatomical_segment_name(skeleton_segment_name=name)
        for name in skeleton.segments
    }


def distribute_segment_masses(
    *,
    skeleton: SkeletonDefinition,
    body_mass: float,
    anthropometric: AnthropometricParameters,
) -> dict[str, float]:
    """Distribute body mass to each skeleton segment, length-cubed within an anatomical segment.

    Each anatomical segment contributes its de Leva mass fraction times body mass (per
    side for bilateral segments). Within an anatomical segment - and within a side of a
    bilateral one - that mass is split across its skeleton segments proportionally to the
    cube of each segment's length, which is the equal-density approximation. The total
    over every skeleton segment is body_mass.
    """
    if body_mass <= 0.0:
        raise ValueError(f"body_mass must be positive, got {body_mass}")

    mapping = map_skeleton_segments(skeleton=skeleton)

    groups: dict[tuple[str, str | None], list[str]] = {}
    for segment_name in skeleton.segments:
        anatomical = mapping[segment_name]
        segment = anthropometric.get(name=anatomical)
        side = _side_of(segment_name=segment_name) if segment.bilateral else None
        groups.setdefault((anatomical, side), []).append(segment_name)

    masses: dict[str, float] = {}
    for (anatomical, _side), segment_names in groups.items():
        total = anthropometric.get(name=anatomical).mass_fraction * body_mass
        length_cubed = {
            name: skeleton.segments[name].length**3 for name in segment_names
        }
        denominator = sum(length_cubed.values())
        for name in segment_names:
            masses[name] = total * length_cubed[name] / denominator
    return masses


def _side_of(*, segment_name: str) -> str | None:
    if segment_name.startswith("left_"):
        return "left"
    if segment_name.startswith("right_"):
        return "right"
    return None
