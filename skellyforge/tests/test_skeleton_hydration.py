"""Tests for closed-form skeleton hydration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton_parts.pose.hydration import (
    hydrate_segment,
    hydrate_skeleton,
)

SKELETON_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)
REST_POSE_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "rest_pose.yaml"
)


def _landmark(name: str, position: tuple[float, float, float], segment: str) -> AnatomicalLandmark:
    return AnatomicalLandmark(
        name=name,
        anatomical_definition=f"the {name}",
        local_position=Point.from_xyz(x=position[0], y=position[1], z=position[2]),
        segment=segment,
    )


def _tetrahedron_segment() -> RigidBodySegment:
    """Four non-coplanar landmarks: enough for a full rigid fit."""
    return RigidBodySegment(
        name="tetra",
        landmarks={
            "origin": _landmark("origin", (0.0, 0.0, 0.0), "tetra"),
            "x": _landmark("x", (10.0, 0.0, 0.0), "tetra"),
            "y": _landmark("y", (0.0, 10.0, 0.0), "tetra"),
            "z": _landmark("z", (0.0, 0.0, 10.0), "tetra"),
        },
        frame_definition=ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=SpatialAxis.Y,
            primary_point_name="y",
            secondary_axis=SpatialAxis.X,
            secondary_point_name="x",
        ),
    )


def _two_landmark_segment() -> RigidBodySegment:
    return RigidBodySegment(
        name="upper_arm",
        landmarks={
            "shoulder": _landmark("shoulder", (0.0, 0.0, 0.0), "upper_arm"),
            "elbow": _landmark("elbow", (0.0, 300.0, 0.0), "upper_arm"),
        },
        frame_definition=ReferenceFrameDefinition(
            origin_point_name="shoulder",
            primary_axis=SpatialAxis.Y,
            primary_point_name="elbow",
        ),
    )


def test_hydrate_segment_rigid_fits_many_landmarks() -> None:
    segment = _tetrahedron_segment()
    expected = Transform(
        rotation=RotationQuaternion.from_components(w=0.5, x=0.5, y=0.5, z=0.5),
        translation=Displacement.from_xyz(x=1.0, y=2.0, z=3.0),
    )
    observed = {
        name: expected.apply(points=landmark.local_position)
        for name, landmark in segment.landmarks.items()
    }

    pose = hydrate_segment(segment=segment, observed=observed)

    np.testing.assert_allclose(pose.origin.array, [1.0, 2.0, 3.0], atol=1e-8)
    for name, landmark in segment.landmarks.items():
        np.testing.assert_allclose(
            pose.origin.array + pose.orientation.rotate_vector(vector=landmark.local_position.array),
            observed[name].array,
            atol=1e-8,
        )


def test_hydrate_segment_recovers_a_direction_for_two_landmarks() -> None:
    segment = _two_landmark_segment()
    # The arm is observed pointing along +x instead of its authored +y.
    observed = {
        "shoulder": Point.from_xyz(x=0.0, y=0.0, z=0.0),
        "elbow": Point.from_xyz(x=300.0, y=0.0, z=0.0),
    }
    pose = hydrate_segment(segment=segment, observed=observed)
    np.testing.assert_allclose(
        pose.orientation.rotate_vector(vector=np.array([0.0, 1.0, 0.0])),
        [1.0, 0.0, 0.0],
        atol=1e-8,
    )
    np.testing.assert_allclose(pose.origin.array, [0.0, 0.0, 0.0], atol=1e-8)


def test_hydrate_segment_raises_without_enough_landmarks() -> None:
    segment = _two_landmark_segment()
    observed = {"shoulder": Point.from_xyz(x=0.0, y=0.0, z=0.0)}
    with pytest.raises(ValueError, match="cannot hydrate"):
        hydrate_segment(segment=segment, observed=observed)


def test_hydrate_skeleton_recovers_the_rest_pose() -> None:
    from skellyforge.core.skeleton_parts.pose.rest_pose import RestPose
    from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition

    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)

    hydrated = hydrate_skeleton(skeleton=skeleton, observed=rest_pose.landmark_positions)

    assert len(hydrated.segment_poses) == 61
    for name, pose in hydrated.segment_poses.items():
        np.testing.assert_allclose(
            pose.origin.array, rest_pose.segment_origins[name].array, atol=1e-6
        )

    # The skull (many landmarks, rigid fit) recovers the rest pose orientation exactly.
    skull_orientation = hydrated.segment_poses["skull"].orientation
    expected = rest_pose.segment_orientations["skull"]
    assert abs(skull_orientation.dot(other=expected)) > 0.999
