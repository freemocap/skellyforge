"""Joint frame batches retain conventions, missingness, and input provenance."""

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.linkage.joint_batch import JointBatchGeometry
from skellyforge.core.skeleton.linkage.joint_pose import compute_joint_poses
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SegmentPose, SkeletonPose
from skellyforge.tests.test_anchored_batch import batch_fixture


def test_joint_batches_match_scalar_with_missing_segments_and_chunk_boundaries():
    skeleton, _, _, rotations, present, _ = batch_fixture(frames=37)
    geometry = JointBatchGeometry.from_skeleton(skeleton=skeleton)
    solutions = np.empty(present.shape, dtype=object)
    for i, solution in enumerate(PoseSolution):
        solutions[i::4] = solution
    result = geometry.calculate(world_orientations=rotations, segment_present=present, segment_solutions=solutions)
    for frame in range(len(rotations)):
        pose = SkeletonPose(segment_poses={name: SegmentPose(segment_name=name,
            origin=Point.from_xyz(x=0., y=0., z=0.),
            orientation=RotationQuaternion(*map(float, rotations[frame, i])),
            scale_estimate=1., solved_by=solutions[frame, i])
            for i, name in enumerate(geometry.segment_names) if present[frame, i]})
        expected = compute_joint_poses(skeleton=skeleton, pose=pose)
        for i, name in enumerate(geometry.joint_names):
            assert result.joint_present[frame, i] == (name in expected)
            if name in expected:
                np.testing.assert_allclose(result.relative_orientations[frame, i], expected[name].relative_orientation.as_array(), atol=1e-15, rtol=0)
                np.testing.assert_allclose(result.angles[frame, i], expected[name].angles, atol=1e-12, rtol=0)
                assert geometry.angle_names[i] == expected[name].angle_names
                assert result.parent_solutions[frame, i] is expected[name].provenance.parent_solution
                assert result.child_solutions[frame, i] is expected[name].provenance.child_solution
            else:
                assert np.isnan(result.angles[frame, i]).all()
                assert result.parent_solutions[frame, i] is None
    for size in [1, 8, 32]:
        chunks = [geometry.calculate(world_orientations=rotations[i:i+size], segment_present=present[i:i+size],
            segment_solutions=solutions[i:i+size]).angles for i in range(0, len(rotations), size)]
        np.testing.assert_array_equal(np.concatenate(chunks), result.angles)
    assert geometry.calculate(world_orientations=rotations[:0], segment_present=present[:0], segment_solutions=solutions[:0]).angles.shape == (0, len(geometry.joint_names), 3)


def test_joint_batch_rejects_unknown_provenance_and_invalid_rotations():
    skeleton, _, _, rotations, present, _ = batch_fixture()
    geometry = JointBatchGeometry.from_skeleton(skeleton=skeleton)
    solutions = np.full(present.shape, None, dtype=object)
    with pytest.raises(ValueError, match="provenance"):
        geometry.calculate(world_orientations=rotations, segment_present=present, segment_solutions=solutions)
    solutions[:] = PoseSolution.DIRECTION
    rotations[2] *= 2
    with pytest.raises(ValueError, match="unit"):
        geometry.calculate(world_orientations=rotations, segment_present=present, segment_solutions=solutions)
