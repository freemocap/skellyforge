"""Per-segment length calibration from observed landmark positions.

The authored lengths are anatomical estimates for a 50th-percentile adult; the real
subject's lengths come from the data. Each segment's length is the distance between its
origin and primary landmarks, so it is estimated as the median of that distance across the
observed frames - the median rather than the mean because a handful of badly triangulated
frames should not stretch a bone.

A segment whose origin or primary landmark is missing cannot be measured, and this refuses
to measure it rather than quietly leaving it out of the result. A caller working with
partial data says so, by passing only the segments it expects to be measurable, or by
asking `measurable_segments` which those are.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName


def estimate_segment_lengths(
    *, segments: Sequence[RigidBodySegment], observed: Mapping[LandmarkNameString, Point]
) -> dict[RigidBodySegmentName, float]:
    """Estimate each segment's length as the median observed origin-to-primary distance.

    Works over a single frame (each Point of shape `(3,)`) or a whole take (shape
    `(num_frames, 3)`); the median is taken over the leading axis.

    Args:
        segments: the segments to measure. Every one of them must have both its origin and
            its primary landmark in `observed`.
        observed: observed landmark positions, keyed by landmark name.

    Returns:
        A mapping of segment name to estimated length, in the units `observed` is in -
        millimetres, for anything coming from these definitions.

    Raises:
        ValueError: a segment's origin or primary landmark is absent from `observed`. A
            short result dictionary would otherwise look exactly like a skeleton with
            fewer segments, and a caller comparing estimated lengths against authored ones
            would silently compare a subset.
    """
    unmeasurable = {
        segment.name: sorted(_missing_landmarks(segment=segment, observed=observed))
        for segment in segments
        if _missing_landmarks(segment=segment, observed=observed)
    }
    if unmeasurable:
        raise ValueError(
            "cannot estimate lengths - these segments' origin or primary landmarks are "
            f"not in `observed`: {sorted(unmeasurable.items())}. Pass only the segments "
            "you expect to measure (see `measurable_segments`) if the data is partial."
        )

    lengths: dict[RigidBodySegmentName, float] = {}
    for segment in segments:
        displacement = (
            observed[segment.frame_definition.primary_point_name]
            - observed[segment.frame_definition.origin_point_name]
        )
        lengths[segment.name] = float(np.median(displacement.norm()))
    return lengths


def measurable_segments(
    *, segments: Sequence[RigidBodySegment], observed: Mapping[LandmarkNameString, Point]
) -> list[RigidBodySegment]:
    """The subset of `segments` whose origin and primary landmarks are both observed."""
    return [
        segment
        for segment in segments
        if not _missing_landmarks(segment=segment, observed=observed)
    ]


def _missing_landmarks(
    *, segment: RigidBodySegment, observed: Mapping[LandmarkNameString, Point]
) -> set[LandmarkNameString]:
    """Which of the two landmarks a segment's length needs are absent from `observed`."""
    return {
        segment.frame_definition.origin_point_name,
        segment.frame_definition.primary_point_name,
    } - set(observed)
