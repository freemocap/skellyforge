"""Incomplete evidence must not produce incomplete modeled anatomy."""
import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, SegmentPose, PoseSolution
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.completion import complete_pose


@pytest.fixture
def setup():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    root = SegmentPose(rest.root_segment_name, Point.from_xyz(x=123., y=-40., z=900.),
        RotationQuaternion.from_rotation_vector(rotation_vector=np.array([.2, -.1, .4])),
        1700., PoseSolution.RIGID_FIT)
    return skeleton, rest, root, dict.fromkeys(skeleton.segments, 1700.)


def test_root_only_emits_every_segment_and_every_landmark_at_relative_rest(setup):
    skeleton, rest, root, scales = setup
    measured = SkeletonPose({root.segment_name: root})
    result = complete_pose(skeleton=skeleton, rest=rest, observed=measured, segment_scales=scales)
    assert set(result.segment_origins) == set(skeleton.segments)
    assert set(result.landmarks) == set(skeleton.landmarks)
    assert result.inferred_segments == set(skeleton.segments) - {root.segment_name}
    assert len(measured.segment_poses) == 1
    for name, parent in rest.parents.items():
        if parent is not None:
            relative = result.segment_rotations_world[parent].inverse() * result.segment_rotations_world[name]
            np.testing.assert_allclose(relative.as_array(), rest.relative_orientations[name].as_array(), atol=1e-12)
            np.testing.assert_allclose(result.segment_origins[name].array, result.landmarks[rest.connect_ats[name]].array)
    for name, landmark in skeleton.landmarks.items():
        expected = result.segment_origins[landmark.segment].array + result.segment_rotations_world[landmark.segment].rotate_vector(
            vector=landmark.local_position.array * scales[landmark.segment])
        np.testing.assert_allclose(result.landmarks[name].array, expected)


def test_wholly_unobserved_pose_needs_explicit_root_to_exist(setup):
    skeleton, rest, root, scales = setup
    args = dict(skeleton=skeleton, rest=rest, observed=SkeletonPose({}), segment_scales=scales)
    assert complete_pose(**args) is None
    result = complete_pose(**args, root_seed=root)
    assert result.inferred_segments == set(skeleton.segments)


def test_observed_child_orientation_is_retained_when_parent_is_inferred(setup):
    skeleton, rest, root, scales = setup
    name = 'left_foot'
    child = SegmentPose(name, Point.from_xyz(x=0., y=0., z=0.), RotationQuaternion.identity(), 1700., PoseSolution.RIGID_FIT)
    args = dict(skeleton=skeleton, rest=rest, observed=SkeletonPose({name: child}), segment_scales=scales)
    with pytest.raises(ValueError, match='root'):
        complete_pose(**args)
    result = complete_pose(**args, root_seed=root)
    np.testing.assert_allclose(result.segment_rotations_world[name].as_array(), child.orientation.as_array())
    assert name not in result.inferred_segments
    assert set(result.landmarks) == set(skeleton.landmarks)
