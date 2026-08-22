"""Per-segment length calibration from observed landmark positions.

The authored lengths are anatomical estimates; the real subject's lengths come from the data.
Each segment's length is the distance between its origin and primary landmarks, so it is
estimated as the median of that distance across the observed frames.
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

    Works over a single frame (each Point of shape (3,)) or a whole take (shape
    (num_frames, 3)); the median is taken over the leading axis. A segment whose origin or
    primary landmark is absent from observed is skipped, since its length cannot be measured.

    Args:
        segments: the segments to measure.
        observed: observed landmark positions, keyed by landmark name.

    Returns:
        A mapping of segment name to estimated length (millimetres).
    """
    lengths: dict[RigidBodySegmentName, float] = {}
    for segment in segments:
        origin_name = segment.frame_definition.origin_point_name
        primary_name = segment.frame_definition.primary_point_name
        if origin_name not in observed or primary_name not in observed:
            continue
        displacement = observed[primary_name] - observed[origin_name]
        lengths[segment.name] = float(np.median(displacement.norm()))
    return lengths
