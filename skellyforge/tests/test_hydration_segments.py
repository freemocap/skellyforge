"""Prepared segment batches preserve the independent scalar hydration contract."""

from dataclasses import fields
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.pose import hydration_batch
from skellyforge.core.skeleton.pose.hydration import (
    hydrate_segment,
    MissingLandmarkObservations,
    DegenerateObservations,
)
from skellyforge.core.skeleton.pose.hydration_batch import LandmarkPoseGeometry
from skellyforge.core.skeleton.pose.pose_tensor import HydrationStatus, decode_solutions
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@pytest.fixture
def mixed_geometry():
    """Repeated conventions with different point names, origins and marker layouts."""
    axis_pairs = [
        (primary, secondary)
        for primary, secondary in product(SpatialAxis, repeat=2)
        if primary.index != secondary.index
    ]
    definitions = [("observation", pair) for pair in axis_pairs for _ in range(2)]
    definitions += [("direction", (axis, None)) for axis in SpatialAxis]
    definitions += [("rigid", (SpatialAxis.X, SpatialAxis.Y))] * 2
    segments, landmarks = {}, {}
    for index, (solver_kind, (primary_axis, secondary_axis)) in enumerate(definitions):
        segment_name = f"segment_{index}"
        local_values = {
            "segment_origin": np.zeros(3),
            "segment_primary": np.array([1.4, 0.2, 0.1]),
        }
        if solver_kind == "observation":
            local_values.update(
                frame_origin=np.array([0.3, 0.1, -0.4]),
                frame_primary=np.array([0.8, 0.3, 0.1]),
                frame_secondary=np.array([0.1, 0.9, 0.5]),
            )
        elif solver_kind == "rigid":
            local_values.update(
                plane_marker=np.array([0.1, 0.8, 0.3]),
                line_marker=2 * local_values["segment_primary"],
            )
        point_names = {role: f"{segment_name}_{role}" for role in local_values}
        owned = {
            point_names[role]: AnatomicalLandmark(
                name=point_names[role],
                anatomical_definition="Synthetic hydration marker",
                local_position=Point(value),
                segment=segment_name,
            )
            for role, value in local_values.items()
        }
        frame_definition = ReferenceFrameDefinition(
            origin_point_name=point_names["segment_origin"],
            primary_point_name=point_names["segment_primary"],
            primary_axis=primary_axis,
        )
        observation_frame = None
        if solver_kind == "observation":
            observation_frame = ReferenceFrameDefinition(
                origin_point_name=point_names["frame_origin"],
                primary_point_name=point_names["frame_primary"],
                secondary_point_name=point_names["frame_secondary"],
                primary_axis=primary_axis,
                secondary_axis=secondary_axis,
            )
        segments[segment_name] = RigidBodySegment(
            name=segment_name,
            landmarks=owned,
            frame_definition=frame_definition,
            observation_frame=observation_frame,
        )
        landmarks.update(owned)
    skeleton = SkeletonDefinition(
        name="mixed_geometry", landmarks=landmarks, segments=segments
    )
    names = tuple(
        reversed(landmarks)
    )  # Input column order must not become output segment order.
    geometry = LandmarkPoseGeometry.from_skeleton(
        skeleton=skeleton, landmark_names=names
    )
    rotation = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.4, -0.7, 0.2])
    ).to_rotation_matrix()
    positions = np.stack(
        [
            rotation @ landmarks[name].local_position.array * 37 + [10.0, 30.0, -20.0]
            for name in names
        ]
    )
    positions = np.repeat(positions[None], 7, axis=0)
    present = np.ones(positions.shape[:2], dtype=bool)
    present[1] = False
    positions[2] = 0.0
    present[3] = np.random.default_rng(345).random(len(names)) > 0.2
    for segment in segments.values():
        if segment.observation_frame is not None:
            frame = segment.observation_frame
            present[4, names.index(frame.secondary_point_name)] = False
            origin, primary, secondary = (
                names.index(name) for name in frame.point_names
            )
            positions[5, secondary] = 2 * positions[5, primary] - positions[5, origin]
        elif segment.supports_rigid_fit:
            for name in segment.landmarks:
                if name.endswith(("plane_marker", "line_marker")):
                    present[4, names.index(name)] = False
                if name.endswith("plane_marker"):
                    present[5, names.index(name)] = False
        else:
            origin = names.index(segment.frame_definition.origin_point_name)
            primary = names.index(segment.frame_definition.primary_point_name)
            positions[6, origin] = 0.0
            positions[6, primary] = (
                -segment.landmarks[names[primary]].local_position.array * 37
            )
    positions[~present] = np.nan
    return geometry, positions, present


def assert_scalar_parity(geometry, positions, present, actual):
    solutions = decode_solutions(actual.solution_codes)
    for frame_index in range(len(positions)):
        observed = {
            name: Point(positions[frame_index, index])
            for index, name in enumerate(geometry.landmark_names)
            if present[frame_index, index]
        }
        for segment_index, segment in enumerate(geometry.skeleton.segments.values()):
            try:
                expected = hydrate_segment(segment=segment, observed=observed)
            except MissingLandmarkObservations:
                assert (
                    actual.hydration_status[frame_index, segment_index]
                    == HydrationStatus.MISSING
                )
            except DegenerateObservations:
                assert (
                    actual.hydration_status[frame_index, segment_index]
                    == HydrationStatus.DEGENERATE
                )
            else:
                assert (
                    actual.hydration_status[frame_index, segment_index]
                    == HydrationStatus.VALID
                )
                np.testing.assert_allclose(
                    actual.origins[frame_index, segment_index],
                    expected.origin.array,
                    atol=1e-11,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    actual.orientations[frame_index, segment_index],
                    expected.orientation.as_array(),
                    atol=1e-13,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    actual.scale_estimates[frame_index, segment_index],
                    expected.scale_estimate,
                    atol=1e-11,
                    rtol=0,
                )
                assert solutions[frame_index, segment_index] is expected.solved_by


def test_segment_groups_match_scalar_signed_axes_and_solver_precedence(mixed_geometry):
    geometry, positions, present = mixed_geometry
    with patch.object(
        hydration_batch,
        "calculate_orthonormal_basis",
        wraps=hydration_batch.calculate_orthonormal_basis,
    ) as basis_calls, patch.object(
        hydration_batch,
        "direction_pair_quaternions",
        wraps=hydration_batch.direction_pair_quaternions,
    ) as direction_calls:
        actual = geometry.recover_tensor(positions=positions, landmark_present=present)
    assert basis_calls.call_count == len(geometry.observation_geometry)
    assert basis_calls.call_count * 2 == sum(
        segment.observation_frame is not None
        for segment in geometry.skeleton.segments.values()
    )
    assert direction_calls.call_count == 1
    assert actual.segment_names == tuple(geometry.skeleton.segments)
    assert_scalar_parity(geometry, positions, present, actual)
    for chunk_size in (1, 3):
        chunks = [
            geometry.recover_tensor(
                positions=positions[index : index + chunk_size],
                landmark_present=present[index : index + chunk_size],
            )
            for index in range(0, len(positions), chunk_size)
        ]
        for array_index, expected in enumerate(actual.arrays()):
            np.testing.assert_allclose(
                np.concatenate([chunk.arrays()[array_index] for chunk in chunks]),
                expected,
                atol=1e-13,
                rtol=0,
            )
    empty = geometry.recover_tensor(
        positions=positions[:0], landmark_present=present[:0], require_all=True
    )
    assert empty.origins.shape == (0, len(geometry.segment_names), positions.shape[-1])
    with pytest.raises(MissingLandmarkObservations, match="Frame 1, segment segment_0"):
        geometry.recover_tensor(
            positions=positions, landmark_present=present, require_all=True
        )
    with pytest.raises(DegenerateObservations, match="Frame 0, segment segment_0"):
        geometry.recover_tensor(
            positions=positions[2:3], landmark_present=present[2:3], require_all=True
        )


def test_preparation_handles_unavailable_columns_and_preserves_retained_tensors(
    mixed_geometry,
):
    original, positions, present = mixed_geometry
    selected = np.arange(len(original.landmark_names)) % 3 != 0
    names = tuple(name for name, keep in zip(original.landmark_names, selected) if keep)
    geometry = LandmarkPoseGeometry.from_skeleton(
        skeleton=original.skeleton, landmark_names=names
    )
    positions, present = positions[:, selected], present[:, selected]
    retained = geometry.recover_tensor(positions=positions, landmark_present=present)
    assert_scalar_parity(geometry, positions, present, retained)
    expected_arrays = tuple(array.copy() for array in retained.arrays())
    later = geometry.recover_tensor(positions=positions * 2, landmark_present=present)
    for retained_array, later_array, expected in zip(
        retained.arrays(), later.arrays(), expected_arrays, strict=True
    ):
        assert not np.shares_memory(retained_array, later_array)
        np.testing.assert_array_equal(retained_array, expected)
    for prepared in (
        *geometry.observation_geometry,
        geometry.direction_geometry,
        *geometry.rigid_geometry,
    ):
        for prepared_field in fields(prepared):
            array = getattr(prepared, prepared_field.name)
            if isinstance(array, np.ndarray):
                with pytest.raises(ValueError):
                    array.flags.writeable = True


def test_human_direction_recovery_uses_one_dispatch_per_call():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest_pose = RestPose.from_default_yaml(skeleton=skeleton)
    names = tuple(rest_pose.landmark_positions)
    geometry = LandmarkPoseGeometry.from_skeleton(
        skeleton=skeleton, landmark_names=names
    )
    positions = (
        np.array([[rest_pose.landmark_positions[name].array for name in names]]) * 1700
    )
    with patch.object(
        hydration_batch,
        "direction_pair_quaternions",
        wraps=hydration_batch.direction_pair_quaternions,
    ) as direction_calls:
        result = geometry.recover_tensor(
            positions=positions, landmark_present=np.ones(positions.shape[:2], bool)
        )
    assert (
        sum(
            segment.observation_frame is None and not segment.supports_rigid_fit
            for segment in skeleton.segments.values()
        )
        > 1
    )
    assert direction_calls.call_count == 1
    assert result.segment_present.all()
