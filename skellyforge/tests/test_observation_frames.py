"""Geometric torso rules, independent of anatomical offsets or rigid-fit residuals."""

from dataclasses import replace

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.skeleton.pose.hydration import (
    hydrate_segment,
    MissingLandmarkObservations,
    DegenerateObservations,
)
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, PoseSolution


def evidence():
    return {
        n: Point.from_array(values=np.array(p, dtype=float))
        for n, p in {
            "pelvis_origin": [0, 0, 0],
            "left_hip_socket": [-150, 0, 0],
            "right_hip_socket": [150, 0, 0],
            "neck_center": [0, 0, 600],
            "chest_center": [0, 0, 300],
            "left_acromion": [-200, 0, 600],
            "right_acromion": [200, 0, 600],
        }.items()
    }


@pytest.mark.parametrize("name", ["pelvis", "thoracic"])
def test_frame_ignores_offsets_and_follows_rigid_motion(name):
    skeleton = SkeletonDefinition.from_default_yaml()
    segment = skeleton.segments[name]
    points = evidence()
    pose = hydrate_segment(segment=segment, observed=points)
    np.testing.assert_allclose(
        pose.orientation.to_rotation_matrix(), np.eye(3), atol=1e-12
    )
    assert pose.solved_by is PoseSolution.OBSERVATION_FRAME
    q = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.3, -0.7, 0.2])
    )
    translation = np.array([100.0, -200.0, 70.0])
    moved = {
        n: Point.from_array(values=1.8 * q.rotate_vector(vector=p.array) + translation)
        for n, p in points.items()
    }
    # Large unrelated constructed offsets cannot pull the frame or its origin.
    for n in segment.landmarks:
        if n not in moved:
            moved[n] = Point.from_xyz(x=9000.0, y=-5000.0, z=7000.0)
    result = hydrate_segment(segment=segment, observed=moved)
    np.testing.assert_allclose(
        result.orientation.to_rotation_matrix(), q.to_rotation_matrix(), atol=1e-12
    )
    np.testing.assert_allclose(
        result.origin.array,
        1.8 * q.rotate_vector(vector=pose.origin.array) + translation,
    )
    assert result.scale_estimate == pytest.approx(1.8 * pose.scale_estimate)
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    resolver = ContinuousRollResolver.for_skeleton(
        skeleton=skeleton, rest_relative_orientations=rest.relative_orientations
    )
    resolved = resolver.resolve_pose(pose=SkeletonPose(segment_poses={name: result}))
    assert resolved.segment_poses[name].orientation.is_same_rotation(
        other=result.orientation
    )


@pytest.mark.parametrize("name", ["pelvis", "thoracic"])
def test_missing_and_degenerate_defining_points_do_not_fall_back(name):
    segment = SkeletonDefinition.from_default_yaml().segments[name]
    points = evidence()
    del points["neck_center"]
    with pytest.raises(MissingLandmarkObservations):
        hydrate_segment(segment=segment, observed=points)
    points = evidence()
    points["neck_center"] = Point.from_xyz(x=100.0, y=0.0, z=0.0)
    points["left_acromion"] = Point.from_xyz(x=-100.0, y=0.0, z=0.0)
    with pytest.raises(DegenerateObservations):
        hydrate_segment(segment=segment, observed=points)


def test_jitter_has_bounded_rotation_without_roll_spins():
    skeleton = SkeletonDefinition.from_default_yaml()
    rng = np.random.default_rng(42)
    for _ in range(100):
        points = {
            n: Point.from_array(values=p.array + rng.uniform(-1, 1, 3))
            for n, p in evidence().items()
        }
        for name in ("pelvis", "thoracic"):
            pose = hydrate_segment(segment=skeleton.segments[name], observed=points)
            # 1 mm bounded perturbations over hundreds of mm of axis separation.
            assert pose.orientation.angle_to(
                other=RotationQuaternion.identity()
            ) < np.deg2rad(2)


def test_frame_requires_two_axes():
    segment = SkeletonDefinition.from_default_yaml().segments["pelvis"]
    with pytest.raises(ValueError, match="two defining axes"):
        replace(
            segment,
            observation_frame=replace(
                segment.observation_frame,
                secondary_axis=None,
                secondary_point_name=None,
            ),
        )


def test_raised_shoulder_preserves_side_axis_and_projects_up():
    skeleton = SkeletonDefinition.from_default_yaml()
    points = evidence()
    points["left_acromion"] = Point.from_xyz(x=-200.0, y=0.0, z=750.0)
    points["neck_center"] = Point.from_xyz(x=0.0, y=0.0, z=675.0)
    points["chest_center"] = Point.from_xyz(x=0.0, y=0.0, z=337.5)
    pose = hydrate_segment(segment=skeleton.segments["thoracic"], observed=points)
    matrix = pose.orientation.to_rotation_matrix()
    side = np.array([400.0, 0.0, -150.0])
    side /= np.linalg.norm(side)
    up = np.array([0.0, 0.0, 1.0])
    up -= side * np.dot(up, side)
    up /= np.linalg.norm(up)
    np.testing.assert_allclose(matrix[:, 0], side, atol=1e-12)
    np.testing.assert_allclose(matrix[:, 2], up, atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_observation_frame_remains_usable_for_body_alignment():
    from skellyforge.core.biomechanics.body_alignment import BodyReferenceTrack

    skeleton = SkeletonDefinition.from_default_yaml()
    pose = hydrate_segment(segment=skeleton.segments["thoracic"], observed=evidence())
    track = BodyReferenceTrack.from_segment_poses(
        segment_name="thoracic",
        poses=(pose,),
        rest_pose=RestPose.from_default_yaml(skeleton=skeleton),
        timestamps_seconds=np.array([0.0]),
        quality=np.array([0.8]),
    )
    assert track.quality[0] == 0.8
    assert np.isfinite(track.world_from_body).all()
