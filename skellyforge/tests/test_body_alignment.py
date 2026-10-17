"""Body reference estimation with incomplete anatomy and irregular sampling."""

import numpy as np
import pytest
from dataclasses import replace

from skellyforge.core.biomechanics.body_alignment import (
    AlignmentOutcome,
    BodyAlignmentConfig,
    BodyReferenceTrack,
    estimate_body_alignment,
)
from skellyforge.type_overloads import FloatArray
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution


def make_track(*, name: str, times: FloatArray, quality: float) -> BodyReferenceTrack:
    return BodyReferenceTrack(
        segment_name=name,
        timestamps_seconds=times,
        world_from_body=np.repeat(np.eye(3)[None], len(times), axis=0),
        origins=np.tile(np.array([100.0, 200.0, 300.0]), (len(times), 1)),
        quality=np.full(len(times), quality),
    )


def test_head_only_recovers_rigid_reference() -> None:
    track = make_track(name="skull", times=np.linspace(0, 1, 21), quality=0.9)
    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    track.world_from_body[:] = rotation
    result = estimate_body_alignment(tracks=(track,), config=BodyAlignmentConfig())
    assert result.outcome is AlignmentOutcome.BODY_REFERENCE
    assert result.anchor_segment == "skull"
    assert result.transform is not None
    np.testing.assert_allclose(
        result.transform.rotation.to_rotation_matrix() @ rotation, np.eye(3), atol=1e-12
    )
    np.testing.assert_allclose(
        result.transform.rotation.to_rotation_matrix() @ track.origins[0]
        + result.transform.translation.array,
        np.zeros(3),
        atol=1e-12,
    )


def test_head_quality_wins_over_denser_torso_track() -> None:
    head = make_track(name="skull", times=np.linspace(0, 1, 11), quality=0.95)
    torso = make_track(name="thorax", times=np.linspace(0, 1, 101), quality=0.6)
    result = estimate_body_alignment(tracks=(torso, head), config=BodyAlignmentConfig())
    assert result.anchor_segment == "skull"


def test_missing_samples_and_long_gaps_cannot_supply_dwell() -> None:
    track = make_track(
        name="skull", times=np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2]), quality=0.9
    )
    result = estimate_body_alignment(tracks=(track,), config=BodyAlignmentConfig())
    assert result.outcome is AlignmentOutcome.INSUFFICIENT_EVIDENCE
    assert result.transform is None
    continuous = make_track(name="skull", times=np.linspace(0, 1, 11), quality=0.9)
    continuous.quality[::3] = 0
    assert (
        estimate_body_alignment(
            tracks=(continuous,), config=BodyAlignmentConfig()
        ).transform
        is None
    )


def test_irregular_sampling_uses_elapsed_time() -> None:
    track = make_track(
        name="skull",
        times=np.array([0.0, 0.01, 0.02, 0.2, 0.3, 0.45, 0.6]),
        quality=0.9,
    )
    result = estimate_body_alignment(tracks=(track,), config=BodyAlignmentConfig())
    assert result.support_seconds == pytest.approx(0.6)


def test_rotating_head_does_not_create_stable_reference() -> None:
    track = make_track(name="skull", times=np.linspace(0, 1, 21), quality=0.9)
    for index, angle in enumerate(np.linspace(0, np.pi, 21)):
        track.world_from_body[index] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    assert (
        estimate_body_alignment(tracks=(track,), config=BodyAlignmentConfig()).transform
        is None
    )


def test_invalid_rotation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        BodyReferenceTrack(
            segment_name="skull",
            timestamps_seconds=np.array([0.0, 1.0]),
            world_from_body=np.zeros((2, 3, 3)),
            origins=np.zeros((2, 3)),
            quality=np.ones(2),
        )


def test_model_pose_adapter_removes_authored_rest_orientation() -> None:
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    rotation = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.3, -0.2, 0.7])
    )
    translation = np.array([100.0, 200.0, 300.0])
    observed = {
        name: Point.from_array(
            values=rotation.rotate_vector(vector=point.array * 1700.0) + translation
        )
        for name, point in rest.landmark_positions.items()
    }
    hydrated = hydrate_skeleton(skeleton=skeleton, observed=observed, require_all=False)
    for name, pose in hydrated.segment_poses.items():
        if pose.solved_by is not PoseSolution.RIGID_FIT:
            continue
        track = BodyReferenceTrack.from_segment_poses(
            segment_name=name,
            poses=(pose,) * 11,
            rest_pose=rest,
            timestamps_seconds=np.linspace(0, 1, 11),
            quality=np.full(11, 0.9),
        )
        np.testing.assert_allclose(
            track.world_from_body[0], rotation.to_rotation_matrix(), atol=1e-8
        )
    skull = hydrated.segment_poses["skull"]
    mixed = BodyReferenceTrack.from_segment_poses(
        segment_name=skull.segment_name,
        poses=(
            skull,
            None,
            replace(skull, solved_by=PoseSolution.DIRECTION),
            replace(skull, solved_by=PoseSolution.TRANSPORTED_ROLL),
        ),
        rest_pose=rest,
        timestamps_seconds=np.arange(4, dtype=np.float64),
        quality=np.ones(4),
    )
    np.testing.assert_array_equal(mixed.quality, np.array([1.0, 0.0, 0.0, 0.0]))
