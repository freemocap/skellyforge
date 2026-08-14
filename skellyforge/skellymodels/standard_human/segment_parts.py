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

from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    SegmentDefinition,
)


@dataclass(frozen=True)
class SegmentPart:
    """A named group of segments, authored without a side."""

    name: str
    segments: tuple[SegmentDefinition, ...]


def _prefixed(value: str | None, prefix: str) -> str | None:
    return None if value is None else f"{prefix}{value}"


def _prefixed_axis(
    axes: tuple[AxisDefinition, ...], prefix: str
) -> tuple[AxisDefinition, ...]:
    return tuple(
        AxisDefinition(
            axis=a.axis,
            kind=a.kind,
            target_keypoint=f"{prefix}{a.target_keypoint}",
        )
        for a in axes
    )


def instantiate_part(part: SegmentPart, prefix: str) -> list[SegmentDefinition]:
    """Expand one part under *prefix*, prefixing names, keypoints and parents."""
    return [
        dataclasses.replace(
            segment,
            name=f"{prefix}{segment.name}",
            parent=_prefixed(segment.parent, prefix),
            rigid_points=tuple(f"{prefix}{p}" for p in segment.rigid_points),
            origin_keypoint=f"{prefix}{segment.origin_keypoint}",
            axes=_prefixed_axis(segment.axes, prefix),
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

    References resolve by **name agreement**, parents and keypoints alike. A
    prefixed reference that does not exist falls back to its unprefixed name when
    that name is declared (a midline segment or keypoint): ``left_shoulder`` finds
    the midline ``upper_chest``, its twist reference finds ``neck_center``. A
    reference whose fallback resolves to nothing is kept as authored — the
    composed model's validators (Task 4) raise on unresolvable parents.
    """
    midline_keypoints: set[str] = set()
    for part, prefix in parts:
        if prefix == "":
            for segment in part.segments:
                midline_keypoints |= segment.required_keypoints()

    composed: list[SegmentDefinition] = []
    seen: set[str] = set()
    for part, prefix in parts:
        for segment in instantiate_part(part, prefix):
            if segment.name in seen:
                raise ValueError(
                    f"duplicate segment name {segment.name!r} after composing part "
                    f"{part.name!r} with prefix {prefix!r}"
                )
            if prefix:
                segment = _resolve_midline_references(segment, midline_keypoints)
            seen.add(segment.name)
            composed.append(segment)

    resolved: list[SegmentDefinition] = []
    for segment in composed:
        parent = segment.parent
        if parent is not None and parent not in seen:
            unprefixed = parent.split("_", 1)[1] if "_" in parent else parent
            if unprefixed in seen:
                parent = unprefixed
        resolved.append(dataclasses.replace(segment, parent=parent))
    return resolved


def _resolve_midline_references(
    segment: SegmentDefinition,
    midline_keypoints: set[str],
) -> SegmentDefinition:
    """Fall prefixed keypoint references back to midline names where they exist.

    ``left_neck_center`` does not exist — ``neck_center`` does. References that do
    not name a midline keypoint (``left_elbow``) are left prefixed.
    """
    def resolved(name: str | None) -> str | None:
        if name is None:
            return None
        if name in midline_keypoints:
            return name
        unprefixed = name.split("_", 1)[1] if "_" in name else name
        return unprefixed if unprefixed in midline_keypoints else name

    return dataclasses.replace(
        segment,
        rigid_points=tuple(resolved(p) for p in segment.rigid_points),
        origin_keypoint=resolved(segment.origin_keypoint),
        axes=tuple(
            AxisDefinition(
                axis=a.axis,
                kind=a.kind,
                target_keypoint=resolved(a.target_keypoint),
            )
            for a in segment.axes
        ),
    )
