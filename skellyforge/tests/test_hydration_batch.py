"""Frame-batched pose recovery retains scalar solver precedence and missingness."""

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion, rotation_vectors_to_quaternions, rotation_matrices_to_quaternions
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import direction_pair_quaternions, rotation_between_vectors
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton, MissingLandmarkObservations
from skellyforge.core.skeleton.pose.hydration_batch import LandmarkPoseGeometry
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def test_quaternion_batch_conversions_cover_branches_and_singular_directions():
    rng = np.random.default_rng(342)
    vectors = np.concatenate([rng.normal(size=(100, 3)) * 3, np.eye(3)*np.pi, np.zeros((1, 3)), np.ones((1, 3))*1e-15])
    expected = [RotationQuaternion.from_rotation_vector(rotation_vector=v) for v in vectors]
    np.testing.assert_allclose(rotation_vectors_to_quaternions(vectors=vectors), [q.as_array() for q in expected], atol=1e-15, rtol=0)
    matrices = np.asarray([q.to_rotation_matrix() for q in expected])
    np.testing.assert_allclose(rotation_matrices_to_quaternions(matrices=matrices),
        [RotationQuaternion.from_rotation_matrix(matrix=m).as_array() for m in matrices], atol=1e-15, rtol=0)
    source = np.concatenate([np.eye(3), np.eye(3), np.eye(3)])
    target = np.concatenate([np.eye(3), -np.eye(3), np.roll(np.eye(3), 1, axis=0)])
    actual = direction_pair_quaternions(from_directions=UnitVector(source), to_directions=UnitVector(target))
    expected = [rotation_between_vectors(from_direction=UnitVector(a), to_direction=UnitVector(b)).as_array() for a, b in zip(source, target)]
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


def test_hydration_matches_scalar_for_random_missing_and_degenerate_frames():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    names = tuple(rest.landmark_positions)
    geometry = LandmarkPoseGeometry.from_skeleton(skeleton=skeleton, landmark_names=names)
    positions = np.broadcast_to(np.asarray([rest.landmark_positions[n].array for n in names]) * 1700, (12, len(names), 3)).copy()
    rng = np.random.default_rng(57)
    positions[2:] += rng.normal(size=positions[2:].shape) * .2
    present = rng.random(positions.shape[:2]) > .15
    present[0] = True
    present[1] = True
    positions[1] = 0.  # Explicitly observed degeneracy, not missing evidence.
    present[2] = False
    actual = geometry.recover(positions=positions, landmark_present=present)
    for frame in range(len(positions)):
        pose = hydrate_skeleton(skeleton=skeleton, observed={n: Point.from_array(values=positions[frame, i]) for i, n in enumerate(names) if present[frame, i]}, require_all=False)
        assert set(pose.segment_poses) == {n for i, n in enumerate(geometry.segment_names) if actual.segment_present[frame, i]}
        for i, name in enumerate(geometry.segment_names):
            if name in pose.segment_poses:
                expected = pose.segment_poses[name]
                np.testing.assert_allclose(actual.origins[frame, i], expected.origin.array, atol=1e-10, rtol=0)
                np.testing.assert_allclose(actual.orientations[frame, i], expected.orientation.as_array(), atol=1e-12, rtol=0)
                np.testing.assert_allclose(actual.scale_estimates[frame, i], expected.scale_estimate, atol=1e-10, rtol=0)
                assert actual.solutions[frame, i] is expected.solved_by
    assert not actual.segment_present[1].any()
    assert not actual.segment_present[2].any()
    with pytest.raises(MissingLandmarkObservations):
        geometry.recover(positions=positions[2:3], landmark_present=present[2:3], require_all=True)
    empty = geometry.recover(positions=positions[:0], landmark_present=present[:0])
    assert empty.origins.shape[0] == 0
    for size in [1, 5]:
        chunks = [geometry.recover(positions=positions[i:i+size], landmark_present=present[i:i+size]) for i in range(0, len(positions), size)]
        np.testing.assert_allclose(np.concatenate([c.orientations for c in chunks]), actual.orientations, atol=1e-14, rtol=0)


def test_rigid_marker_batches_preserve_subset_fits_and_direction_fallback():
    from skellyforge.tests.test_rigid_marker_skeleton import _board
    from skellyforge.core.skeleton.skeleton_pose import PoseSolution
    from skellyforge.core.skeleton.linkage.joint_batch import JointBatchGeometry
    skeleton = _board()
    names = tuple(skeleton.landmarks)
    geometry = LandmarkPoseGeometry.from_skeleton(skeleton=skeleton, landmark_names=names)
    positions = np.broadcast_to([skeleton.landmarks[n].local_position.array * 54 + [10., 20., 30.] for n in names], (5, len(names), 3)).copy()
    selections = [set(names), {"marker_0", "marker_1", "marker_3"}, {"marker_0", "marker_1"}, {"marker_0", "marker_1", "marker_2"}, set()]
    present = np.asarray([[n in selected for n in names] for selected in selections])
    actual = geometry.recover(positions=positions, landmark_present=present)
    assert actual.solutions[0, 0] is PoseSolution.RIGID_FIT
    assert actual.solutions[1, 0] is PoseSolution.RIGID_FIT
    assert actual.solutions[2, 0] is PoseSolution.DIRECTION
    assert actual.status[3, 0] == 2 and actual.status[4, 0] == 0
    for frame in range(5):
        expected = hydrate_skeleton(skeleton=skeleton, observed={n: Point(positions[frame, i]) for i, n in enumerate(names) if present[frame, i]}, require_all=False)
        assert bool(expected.segment_poses) == bool(actual.segment_present[frame, 0])
        if expected.segment_poses:
            pose = next(iter(expected.segment_poses.values()))
            np.testing.assert_allclose(actual.origins[frame, 0], pose.origin.array, atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual.orientations[frame, 0], pose.orientation.as_array(), atol=1e-14, rtol=0)
            np.testing.assert_allclose(actual.scale_estimates[frame, 0], pose.scale_estimate, atol=1e-12, rtol=0)
    joints = JointBatchGeometry.from_skeleton(skeleton=skeleton).calculate(world_orientations=actual.orientations,
        segment_present=actual.segment_present, segment_solutions=actual.solutions)
    assert joints.angles.shape == (5, 0, 3)
