"""Tests for rotation_between_vectors and primary_axis_unit."""

from __future__ import annotations

import numpy as np

from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import (
    primary_axis_unit,
    rotation_between_vectors,
)


def test_rotation_maps_plus_x_onto_plus_y() -> None:
    rotation = rotation_between_vectors(
        from_direction=UnitVector.from_xyz(x=1.0, y=0.0, z=0.0),
        to_direction=UnitVector.from_xyz(x=0.0, y=1.0, z=0.0),
    )
    np.testing.assert_allclose(
        rotation.rotate_vector(vector=np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-10
    )


def test_rotation_maps_plus_z_onto_plus_x() -> None:
    rotation = rotation_between_vectors(
        from_direction=UnitVector.from_xyz(x=0.0, y=0.0, z=1.0),
        to_direction=UnitVector.from_xyz(x=1.0, y=0.0, z=0.0),
    )
    np.testing.assert_allclose(
        rotation.rotate_vector(vector=np.array([0.0, 0.0, 1.0])), [1.0, 0.0, 0.0], atol=1e-10
    )


def test_rotation_is_identity_for_aligned_vectors() -> None:
    rotation = rotation_between_vectors(
        from_direction=UnitVector.from_xyz(x=0.0, y=1.0, z=0.0),
        to_direction=UnitVector.from_xyz(x=0.0, y=1.0, z=0.0),
    )
    np.testing.assert_allclose(rotation.to_rotation_matrix(), np.eye(3), atol=1e-10)


def test_rotation_flips_an_anti_parallel_vector() -> None:
    rotation = rotation_between_vectors(
        from_direction=UnitVector.from_xyz(x=1.0, y=0.0, z=0.0),
        to_direction=UnitVector.from_xyz(x=-1.0, y=0.0, z=0.0),
    )
    np.testing.assert_allclose(
        rotation.rotate_vector(vector=np.array([1.0, 0.0, 0.0])), [-1.0, 0.0, 0.0], atol=1e-10
    )


def test_primary_axis_unit_returns_the_signed_axis() -> None:
    np.testing.assert_allclose(primary_axis_unit(axis=SpatialAxis.X), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(primary_axis_unit(axis=SpatialAxis.NEGATIVE_Y), [0.0, -1.0, 0.0])
    np.testing.assert_allclose(primary_axis_unit(axis=SpatialAxis.Z), [0.0, 0.0, 1.0])
