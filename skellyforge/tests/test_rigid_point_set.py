"""Tests for the closed-form rigid fit (Kabsch) and the RigidPointSet wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.core.math.kinematics.rigid_point_set import (
    RigidPointSet,
    align_point_sets_kabsch,
)


def test_kabsch_recovers_a_known_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    reference = Point.from_array(values=rng.normal(size=(5, 3)))
    expected = Transform(
        rotation=RotationQuaternion.from_components(w=0.5, x=0.5, y=0.5, z=0.5),
        translation=Displacement.from_xyz(x=10.0, y=-3.0, z=2.0),
    )
    observed = expected.apply(points=reference)

    fit = align_point_sets_kabsch(reference=reference, observed=observed)
    recovered = fit.apply(points=reference)

    np.testing.assert_allclose(recovered.array, observed.array, atol=1e-10)


def test_kabsch_recovers_a_pure_translation() -> None:
    reference = Point.from_array(
        values=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    )
    observed = Point.from_array(
        values=np.array([[5, -2, 3], [6, -2, 3], [5, -1, 3], [5, -2, 4]], dtype=np.float64)
    )
    fit = align_point_sets_kabsch(reference=reference, observed=observed)
    # Identity rotation, translation (5, -2, 3).
    np.testing.assert_allclose(fit.rotation.to_rotation_matrix(), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(fit.translation.array, [5.0, -2.0, 3.0], atol=1e-10)


def test_kabsch_returns_a_proper_rotation_for_a_mirrored_set() -> None:
    # A mirrored point set cannot be reached by any rotation; the fit must still be a
    # proper rotation (determinant +1), never a reflection.
    rng = np.random.default_rng(3)
    reference = Point.from_array(values=rng.normal(size=(6, 3)))
    mirrored = reference.array.copy()
    mirrored[:, 0] *= -1.0
    observed = Point.from_array(values=mirrored)

    fit = align_point_sets_kabsch(reference=reference, observed=observed)
    assert np.linalg.det(fit.rotation.to_rotation_matrix()) > 0.0


def test_kabsch_rejects_collinear_points() -> None:
    reference = Point.from_array(
        values=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    )
    observed = Point.from_array(
        values=np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="collinear"):
        align_point_sets_kabsch(reference=reference, observed=observed)


def test_kabsch_rejects_fewer_than_three_points() -> None:
    reference = Point.from_array(values=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64))
    observed = Point.from_array(values=np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64))
    with pytest.raises(ValueError, match="at least 3"):
        align_point_sets_kabsch(reference=reference, observed=observed)


def test_kabsch_rejects_mismatched_shapes() -> None:
    reference = Point.from_array(values=np.zeros((4, 3)))
    observed = Point.from_array(values=np.zeros((5, 3)))
    with pytest.raises(ValueError, match="same shape"):
        align_point_sets_kabsch(reference=reference, observed=observed)


def test_rigid_point_set_fits_a_pose_from_observed_positions() -> None:
    reference = Point.from_array(
        values=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float64)
    )
    point_set = RigidPointSet(
        point_names=("origin", "x", "y", "z"),
        reference_positions=reference,
    )
    observed = {
        "origin": Point.from_xyz(x=5.0, y=-2.0, z=3.0),
        "x": Point.from_xyz(x=15.0, y=-2.0, z=3.0),
        "y": Point.from_xyz(x=5.0, y=8.0, z=3.0),
        "z": Point.from_xyz(x=5.0, y=-2.0, z=13.0),
    }
    fit = point_set.fit_pose(observed=observed)
    np.testing.assert_allclose(
        fit.apply(points=reference).array,
        np.array([[5, -2, 3], [15, -2, 3], [5, 8, 3], [5, -2, 13]], dtype=np.float64),
        atol=1e-8,
    )


def test_rigid_point_set_fits_with_a_missing_landmark() -> None:
    reference = Point.from_array(
        values=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float64)
    )
    point_set = RigidPointSet(
        point_names=("origin", "x", "y", "z"),
        reference_positions=reference,
    )
    # "z" is missing from the observed set; the fit proceeds on the other three.
    observed = {
        "origin": Point.from_xyz(x=5.0, y=-2.0, z=3.0),
        "x": Point.from_xyz(x=15.0, y=-2.0, z=3.0),
        "y": Point.from_xyz(x=5.0, y=8.0, z=3.0),
    }
    fit = point_set.fit_pose(observed=observed)
    np.testing.assert_allclose(fit.translation.array, [5.0, -2.0, 3.0], atol=1e-8)


def test_rigid_point_set_rejects_too_few_observed_points() -> None:
    reference = Point.from_array(
        values=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float64)
    )
    point_set = RigidPointSet(
        point_names=("a", "b", "c"),
        reference_positions=reference,
    )
    with pytest.raises(ValueError, match="at least 3 observed"):
        point_set.fit_pose(observed={"a": Point.from_xyz(x=0.0, y=0.0, z=0.0)})
