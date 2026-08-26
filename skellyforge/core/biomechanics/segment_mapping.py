"""Mass distribution: skeleton segments → de Leva anatomical segments.

Each skeleton segment declares its de Leva anatomical home in its component
YAML via ``anatomical_segment: <name>``. This module reads those declarations
and distributes body mass across segments proportionally to length cubed
within each anatomical group (the equal-density approximation).
"""

from __future__ import annotations

from collections.abc import Mapping

from skellyforge.core.biomechanics.anthropometric_parameters import AnthropometricParameters
from skellyforge.core.skeleton.loading.sided_expansion import side_prefix_of
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def map_skeleton_segments(*, skeleton: SkeletonDefinition) -> dict[str, str]:
    """Every skeleton segment mapped to its de Leva anatomical segment name.

    Raises:
        ValueError: any segment lacks an ``anatomical_segment`` declaration.
    """
    mapping = {}
    unmapped = []
    for name, segment in skeleton.segments.items():
        if segment.anatomical_segment is None:
            unmapped.append(name)
        else:
            mapping[name] = segment.anatomical_segment
    if unmapped:
        raise ValueError(
            f"these segments have no anatomical_segment declaration in their "
            f"component YAML: {sorted(unmapped)}. Add one so mass distribution "
            f"can proceed."
        )
    return mapping


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
        side = side_prefix_of(name=segment_name) if segment.bilateral else None
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


