"""Existing math needed by future animation exports; no writer or format policy."""

import numpy as np
import pytest

from skellyforge.core.math.geometry.coordinate_systems import (
    CoordinateSystemRegistry, CoordinateSystemTransform,
)
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion, slerp_resample
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SegmentPose, SkeletonPose


def axis_matrix(axis, degrees):
    """Independent elementary matrices, rather than another quaternion round trip."""
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    if axis == "x":
        return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    if axis == "y":
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@pytest.mark.parametrize("angle", [23., 179., 181.])
@pytest.mark.parametrize("sign", [1., -1.])
def test_branching_local_rotations_reconstruct_world_frames(angle, sign):
    parents = {"root": None, "arm": "root", "hand": "arm", "leg": "root"}
    # Noncommuting, asymmetric local frames include authored rest orientation.
    local = {
        "root": axis_matrix("z", angle) @ axis_matrix("x", 17.),
        "arm": axis_matrix("y", -37.) @ axis_matrix("z", 13.),
        "hand": axis_matrix("x", 61.),
        "leg": axis_matrix("z", -29.) @ axis_matrix("y", 42.),
    }
    world = {}
    for name, parent in parents.items():
        world[name] = local[name] if parent is None else world[parent] @ local[name]
    segments = {
        name: SegmentPose(
            segment_name=name, origin=Point.from_xyz(x=120., y=-80., z=900.),
            orientation=RotationQuaternion.from_array(
                array=sign * RotationQuaternion.from_rotation_matrix(matrix=world[name]).as_array()),
            scale_estimate=1700., solved_by=PoseSolution.RIGID_FIT,
        )
        for name in reversed(parents)  # Parent-first storage must not be required.
    }
    pose = SkeletonPose(segment_poses=segments)
    relative = pose.parent_relative_orientations(parents=parents)
    reconstructed = {}
    for name, parent in parents.items():
        rotation = relative[name].to_rotation_matrix()
        np.testing.assert_allclose(rotation, local[name], atol=1e-12)
        reconstructed[name] = rotation if parent is None else reconstructed[parent] @ rotation
        np.testing.assert_allclose(reconstructed[name], world[name], atol=1e-12)
    # A missing parent cannot turn an observed child into a new root or identity.
    partial = SkeletonPose(segment_poses={name: value for name, value in segments.items() if name != "arm"})
    assert set(partial.parent_relative_orientations(parents=parents)) == {"root", "leg"}


@pytest.mark.parametrize("target", ["blender", "vrm", "ros", "unreal", "unity"])
def test_coordinate_conversion_preserves_posed_geometry(target):
    registry = CoordinateSystemRegistry.from_default_yaml()
    conversion = CoordinateSystemTransform(
        from_convention=registry.get(name="blender"), to_convention=registry.get(name=target))
    rotation = axis_matrix("z", 71.) @ axis_matrix("y", -33.)
    origin = np.array([120., -350., 870.])
    # Unequal axes and off-axis attachments expose incorrect order/reflections.
    attachments = np.array([[210., 0., 0.], [15., -43., 9.], [-20., 11., 52.]])
    world = origin + attachments @ rotation.T
    converted_world = conversion.convert_point(point=Point.from_array(values=world)).array
    converted_origin = conversion.convert_point(point=Point.from_array(values=origin)).array
    converted_local = conversion.convert_point(point=Point.from_array(values=attachments)).array
    converted_rotation = conversion.convert_quaternion(
        quaternion=RotationQuaternion.from_rotation_matrix(matrix=rotation)).to_rotation_matrix()
    np.testing.assert_allclose(converted_origin + converted_local @ converted_rotation.T,
                               converted_world, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(converted_world - converted_origin, axis=1),
                               np.linalg.norm(attachments, axis=1), atol=1e-10)


def test_irregular_time_resampling_crosses_half_turn_without_spinning():
    times = np.array([12.0, 12.2, 12.8, 13.0])
    target = np.array([12.0, 12.1, 12.2, 12.5, 12.8, 12.9, 13.0])
    # Known constant angular speed, with deliberately alternating quaternion signs.
    angles = 170. + 20. * (times - times[0])
    half = np.deg2rad(angles) / 2.
    source = np.column_stack((np.cos(half), np.zeros((len(times), 2)), np.sin(half)))
    source[1::2] *= -1.
    original = source.copy()
    actual = slerp_resample(quaternions=source, original_timestamps=times, target_timestamps=target)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-12)
    for timestamp, quaternion in zip(target, actual, strict=True):
        expected = axis_matrix("z", 170. + 20. * (timestamp - times[0]))
        np.testing.assert_allclose(RotationQuaternion.from_array(array=quaternion).to_rotation_matrix(),
                                   expected, atol=1e-10)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(times, [12.0, 12.2, 12.8, 13.0])
