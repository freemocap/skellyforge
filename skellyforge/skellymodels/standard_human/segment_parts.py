"""Parts: named, side-agnostic groups of segments, instantiated by prefix.

A hand has 16 segments and a human has two hands. Writing the hand out twice is
duplicated information, so the hand is authored **once** and instantiated with the
prefixes ``left_`` and ``right_``.

Parts join by **name agreement**: the hand's local ``wrist`` becomes ``left_wrist``
under its prefix, which is already the body's wrist keypoint, and its parent
reference ``lower_arm`` becomes ``left_lower_arm``, which is already a body segment.
There is no separate attachment mechanism — once expanded there is nothing left over
to get wrong.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass

from skellyforge.skellymodels.standard_human.segment_definition import SegmentDefinition


@dataclass(frozen=True)
class SegmentPart:
    """A named group of segments, authored without a side."""

    name: str
    segments: tuple[SegmentDefinition, ...]


def _prefixed(value: str | None, prefix: str) -> str | None:
    return None if value is None else f"{prefix}{value}"


def instantiate_part(part: SegmentPart, prefix: str) -> list[SegmentDefinition]:
    """Expand one part under *prefix*, prefixing names, keypoints and parents."""
    return [
        dataclasses.replace(
            segment,
            name=f"{prefix}{segment.name}",
            parent=_prefixed(segment.parent, prefix),
            origin_keypoint=f"{prefix}{segment.origin_keypoint}",
            long_axis_keypoint=f"{prefix}{segment.long_axis_keypoint}",
            twist_keypoint=_prefixed(segment.twist_keypoint, prefix),
        )
        for segment in part.segments
    ]


def compose_parts(
    parts: Iterable[tuple[SegmentPart, str]],
) -> list[SegmentDefinition]:
    """Expand every (part, prefix) pair into one flat segment list.

    Downstream consumers — the solver, the stream schema — receive a flat indexed
    list exactly as before. They receive it from this build step instead of from a
    file, which is also where the per-frame O(n) lookups stop being O(n^2).
    """
    composed: list[SegmentDefinition] = []
    seen: set[str] = set()
    for part, prefix in parts:
        for segment in instantiate_part(part, prefix):
            if segment.name in seen:
                raise ValueError(
                    f"duplicate segment name {segment.name!r} after composing part "
                    f"{part.name!r} with prefix {prefix!r}"
                )
            seen.add(segment.name)
            composed.append(segment)
    return composed
