"""Connected fitted motion has fixed attachments, independently of measured origins."""

from dataclasses import replace

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@pytest.fixture
def inputs():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    scales = {name: 1400.0 + 11 * index for index, name in enumerate(skeleton.segments)}
    fit = ModelScaleFit(
        fitted_scale=1700.0,
        segment_scales=scales,
        segment_lengths={
            name: segment.length * scales[name]
            for name, segment in skeleton.segments.items()
        },
        measured_segment_names=frozenset(),
        voting_segment_names=frozenset(),
    )
    return skeleton, rest, fit


def test_connected_motion_preserves_fitted_geometry(inputs):
    skeleton, rest, fit = inputs
    originals = {
        name: landmark.local_position.array.copy()
        for name, landmark in skeleton.landmarks.items()
    }
    for frame in range(3):
        rotations = {
            name: RotationQuaternion.from_rotation_vector(
                rotation_vector=np.array([0.13 * frame, 0.02 * index, -0.17])
            )
            for index, name in enumerate(skeleton.segments)
        }
        root = Point.from_xyz(x=100.0 * frame, y=-230.0, z=910.0)
        orientations, origins, landmarks = synthesize_fitted_pose(
            skeleton=skeleton,
            fit=fit,
            segment_relative_orientations={
                name: q
                for name, q in rotations.items()
                if name != rest.root_segment_name
            },
            root_world_orientation=rotations[rest.root_segment_name],
            root_origin=root,
        )
        np.testing.assert_allclose(origins[rest.root_segment_name].array, root.array)
        np.testing.assert_allclose(
            orientations[rest.root_segment_name].to_rotation_matrix(),
            rotations[rest.root_segment_name].to_rotation_matrix(),
            atol=1e-12,
        )
        for joint in skeleton.joints.values():
            parent, child = joint.parent.name, joint.child.name
            parent_matrix = orientations[parent].to_rotation_matrix()
            np.testing.assert_allclose(
                orientations[child].to_rotation_matrix(),
                parent_matrix @ rotations[child].to_rotation_matrix(),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                origins[child].array, landmarks[joint.connect_at.name].array
            )
            np.testing.assert_allclose(
                parent_matrix.T @ (origins[child].array - origins[parent].array),
                originals[joint.connect_at.name] * fit.segment_scales[parent],
                atol=1e-9,
            )
        for name, landmark in skeleton.landmarks.items():
            owner = skeleton.owning_segment_name_of(landmark=landmark)
            np.testing.assert_allclose(
                orientations[owner].to_rotation_matrix().T
                @ (landmarks[name].array - origins[owner].array),
                originals[name] * fit.segment_scales[owner],
                atol=1e-9,
            )
    for name, landmark in skeleton.landmarks.items():
        np.testing.assert_array_equal(landmark.local_position.array, originals[name])


def test_selection_and_missing_motion_are_explicit(inputs):
    skeleton, rest, fit = inputs
    root = rest.root_segment_name
    child = next(
        joint.child.name
        for joint in skeleton.joints.values()
        if joint.parent.name == root
    )
    kwargs = dict(
        skeleton=skeleton,
        fit=fit,
        root_world_orientation=rest.relative_orientations[root],
        root_origin=Point.from_xyz(x=0.0, y=0.0, z=0.0),
        segment_names=frozenset({root, child}),
        segment_relative_orientations={child: rest.relative_orientations[child]},
    )
    orientations, origins, landmarks = synthesize_fitted_pose(**kwargs)
    assert set(orientations) == set(origins) == {root, child}
    assert set(landmarks) == {
        name
        for name, landmark in skeleton.landmarks.items()
        if skeleton.owning_segment_name_of(landmark=landmark) in {root, child}
    }
    with pytest.raises(ValueError, match="Rotations"):
        synthesize_fitted_pose(**(kwargs | {"segment_relative_orientations": {}}))
    with pytest.raises(ValueError, match="root"):
        synthesize_fitted_pose(**(kwargs | {"segment_names": frozenset({child})}))
    grandchild = next(
        joint.child.name
        for joint in skeleton.joints.values()
        if joint.parent.name == child
    )
    with pytest.raises(ValueError, match="ancestor"):
        synthesize_fitted_pose(
            **(
                kwargs
                | {
                    "segment_names": frozenset({root, grandchild}),
                    "segment_relative_orientations": {
                        grandchild: rest.relative_orientations[grandchild]
                    },
                }
            )
        )
    lengths = dict(fit.segment_lengths)
    lengths[child] *= 2
    with pytest.raises(ValueError, match="geometry disagree"):
        synthesize_fitted_pose(
            **(kwargs | {"fit": replace(fit, segment_lengths=lengths)})
        )


def test_uniform_fit_reproduces_scaled_rest_pose(inputs):
    skeleton, rest, fit = inputs
    scale = 1750.0
    fit = replace(
        fit,
        segment_scales={name: scale for name in skeleton.segments},
        segment_lengths={
            name: segment.length * scale for name, segment in skeleton.segments.items()
        },
    )
    root = Point.from_xyz(x=100.0, y=200.0, z=900.0)
    orientations, origins, landmarks = synthesize_fitted_pose(
        skeleton=skeleton,
        fit=fit,
        root_origin=root,
        root_world_orientation=rest.relative_orientations[rest.root_segment_name],
        segment_relative_orientations={
            name: q
            for name, q in rest.relative_orientations.items()
            if name != rest.root_segment_name
        },
    )
    for name in skeleton.segments:
        np.testing.assert_allclose(
            orientations[name].to_rotation_matrix(),
            rest.segment_orientations[name].to_rotation_matrix(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            origins[name].array,
            root.array + scale * rest.segment_origins[name].array,
            atol=1e-9,
        )
    for name in skeleton.landmarks:
        np.testing.assert_allclose(
            landmarks[name].array,
            root.array + scale * rest.landmark_positions[name].array,
            atol=1e-9,
        )
