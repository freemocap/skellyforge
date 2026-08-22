"""Tests for per-segment length estimation."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton_parts.segment_length_estimation import (
    estimate_segment_lengths,
)


def _landmark(name: str, position: tuple[float, float, float]) -> AnatomicalLandmark:
    return AnatomicalLandmark(
        name=name,
        anatomical_definition=f"the {name}",
        local_position=Point.from_xyz(x=position[0], y=position[1], z=position[2]),
        segment="left_upper_arm",
    )


def _upper_arm() -> RigidBodySegment:
    return RigidBodySegment(
        name="left_upper_arm",
        landmarks={
            "shoulder": _landmark("shoulder", (0.0, 0.0, 0.0)),
            "elbow": _landmark("elbow", (0.0, 300.0, 0.0)),
        },
        frame_definition=ReferenceFrameDefinition(
            origin_point_name="shoulder",
            primary_axis=SpatialAxis.Y,
            primary_point_name="elbow",
        ),
    )


def test_estimates_length_from_a_single_frame() -> None:
    segment = _upper_arm()
    observed = {
        "shoulder": Point.from_xyz(x=0.0, y=0.0, z=0.0),
        "elbow": Point.from_xyz(x=0.0, y=310.0, z=0.0),
    }
    lengths = estimate_segment_lengths(segments=[segment], observed=observed)
    assert lengths["left_upper_arm"] == pytest.approx(310.0)


def test_estimates_length_as_the_median_over_frames() -> None:
    segment = _upper_arm()
    observed = {
        "shoulder": Point.from_prevalidated_array(array=np.zeros((5, 3))),
        "elbow": Point.from_prevalidated_array(
            array=np.array(
                [[0, 290, 0], [0, 300, 0], [0, 310, 0], [0, 400, 0], [0, 320, 0]],
                dtype=np.float64,
            )
        ),
    }
    lengths = estimate_segment_lengths(segments=[segment], observed=observed)
    # Median of [290, 300, 310, 400, 320] is 310.
    assert lengths["left_upper_arm"] == pytest.approx(310.0)


def test_skips_a_segment_with_a_missing_landmark() -> None:
    segment = _upper_arm()
    observed = {"shoulder": Point.from_xyz(x=0.0, y=0.0, z=0.0)}
    lengths = estimate_segment_lengths(segments=[segment], observed=observed)
    assert lengths == {}
