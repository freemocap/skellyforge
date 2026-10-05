"""Palm observations orient carpals without treating the knuckles as a rigid set."""
import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_segment, MissingLandmarkObservations, DegenerateObservations
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, PoseSolution


def hand(side):
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    names = ('carpal_origin', 'middle_mcp', 'index_mcp', 'middle_cmc')
    points = {f'{side}_{name}': rest.landmark_positions[f'{side}_{name}'] for name in names}
    return skeleton, rest, points


@pytest.mark.parametrize('side', ['left', 'right'])
def test_carpal_frame_matches_rest_and_follows_motion(side):
    skeleton, rest, points = hand(side)
    name = f'{side}_carpals'
    segment = skeleton.segments[name]
    pose = hydrate_segment(segment=segment, observed=points)
    assert pose.solved_by is PoseSolution.OBSERVATION_FRAME
    assert pose.orientation.is_same_rotation(other=rest.segment_orientations[name])
    assert pose.scale_estimate == pytest.approx(1.)
    q = RotationQuaternion.from_rotation_vector(rotation_vector=np.array([.4, -.6, .7]))
    shift = np.array([200., -340., 120.])
    moved = {n: Point.from_array(values=1700 * q.rotate_vector(vector=p.array) + shift) for n, p in points.items()}
    result = hydrate_segment(segment=segment, observed=moved)
    assert result.orientation.is_same_rotation(other=q * pose.orientation)
    assert result.scale_estimate == pytest.approx(1700.)
    np.testing.assert_allclose(result.origin.array, moved[f'{side}_carpal_origin'].array)
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton, rest_relative_orientations=rest.relative_orientations)
    resolved = resolver.resolve_pose(pose=SkeletonPose(segment_poses={name: result}))
    assert resolved.segment_poses[name].orientation.is_same_rotation(other=result.orientation)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_constructed_points_do_not_steer_orientation_and_knuckles_can_deform(side):
    skeleton, rest, points = hand(side)
    segment = skeleton.segments[f'{side}_carpals']
    original = hydrate_segment(segment=segment, observed=points)
    for name in segment.landmarks:
        if name != f'{side}_carpal_origin':
            points[name] = Point.from_xyz(x=12., y=-8., z=15.)
    changed_offsets = hydrate_segment(segment=segment, observed=points)
    assert changed_offsets.orientation.is_same_rotation(other=original.orientation)
    # Knuckle distance and longitudinal displacement can change without changing
    # the palm plane. No fixed-distance fit is imposed on those observations.
    origin = points[f'{side}_carpal_origin'].array
    longitudinal = points[f'{side}_middle_mcp'].array - origin
    index = points[f'{side}_index_mcp'].array - origin
    points[f'{side}_index_mcp'] = Point.from_array(values=origin + 1.6 * index + .2 * longitudinal)
    deformed = hydrate_segment(segment=segment, observed=points)
    assert deformed.orientation.is_same_rotation(other=original.orientation)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('missing', ['carpal_origin', 'middle_mcp', 'index_mcp', 'middle_cmc'])
def test_missing_frame_input_does_not_fall_back(side, missing):
    skeleton, rest, points = hand(side)
    # Even abundant model-owned points cannot rescue an incomplete observation frame.
    points.update({n: rest.landmark_positions[n] for n in skeleton.segments[f'{side}_carpals'].landmarks})
    del points[f'{side}_{missing}']
    with pytest.raises(MissingLandmarkObservations):
        hydrate_segment(segment=skeleton.segments[f'{side}_carpals'], observed=points)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_collinear_knuckles_do_not_fall_back(side):
    skeleton, rest, points = hand(side)
    points[f'{side}_index_mcp'] = points[f'{side}_middle_mcp']
    with pytest.raises(DegenerateObservations):
        hydrate_segment(segment=skeleton.segments[f'{side}_carpals'], observed=points)
