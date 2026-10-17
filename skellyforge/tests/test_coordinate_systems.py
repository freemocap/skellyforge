"""Tests for coordinate-system conventions and conversion between them."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.coordinate_systems import (
    AnatomicalDirection,
    CoordinateSystemConvention,
    CoordinateSystemRegistry,
    CoordinateSystemTransform,
    conversion_matrix,
    parse_anatomical_direction,
)
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point


def _registry() -> CoordinateSystemRegistry:
    return CoordinateSystemRegistry.from_default_yaml()


def _convention(name: str) -> CoordinateSystemConvention:
    return _registry().get(name=name)


def _random_points(*, seed: int) -> Point:
    return Point.from_array(values=np.random.default_rng(seed).normal(size=(10, 3)))


def test_anatomical_direction_unit_vectors() -> None:
    np.testing.assert_allclose(AnatomicalDirection.RIGHT.unit_vector, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(AnatomicalDirection.FORWARD.unit_vector, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(AnatomicalDirection.UP.unit_vector, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(AnatomicalDirection.LEFT.unit_vector, [-1.0, 0.0, 0.0])


def test_parse_anatomical_direction_is_case_insensitive_and_rejects_unknown() -> None:
    assert parse_anatomical_direction(label="RIGHT") is AnatomicalDirection.RIGHT
    with pytest.raises(ValueError, match="unknown anatomical direction"):
        parse_anatomical_direction(label="diagonal")


@pytest.mark.parametrize("name", ["blender", "vrm", "ros", "isb", "unreal", "unity"])
def test_a_point_round_trips_through_every_convention(name: str) -> None:
    blender = _convention("blender")
    other = _convention(name)
    outward = CoordinateSystemTransform(from_convention=blender, to_convention=other)
    homeward = CoordinateSystemTransform(from_convention=other, to_convention=blender)
    points = _random_points(seed=7)
    round_tripped = homeward.convert_point(point=outward.convert_point(point=points))
    np.testing.assert_allclose(round_tripped.array, points.array, atol=1e-12)


def test_conversion_matrix_determinant_reports_a_handedness_flip() -> None:
    same = conversion_matrix(
        from_convention=_convention("blender"), to_convention=_convention("vrm")
    )
    flipped = conversion_matrix(
        from_convention=_convention("blender"), to_convention=_convention("unreal")
    )
    assert np.linalg.det(same) == pytest.approx(1.0)
    assert np.linalg.det(flipped) == pytest.approx(-1.0)


def test_a_quaternion_round_trips_through_a_handedness_flip() -> None:
    blender = _convention("blender")
    unreal = _convention("unreal")
    outward = CoordinateSystemTransform(from_convention=blender, to_convention=unreal)
    homeward = CoordinateSystemTransform(from_convention=unreal, to_convention=blender)
    quaternion = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.3, -1.2, 0.7])
    )
    round_tripped = homeward.convert_quaternion(
        quaternion=outward.convert_quaternion(quaternion=quaternion)
    )
    np.testing.assert_allclose(
        round_tripped.to_rotation_matrix(), quaternion.to_rotation_matrix(), atol=1e-12
    )


def test_a_displacement_round_trips_and_keeps_its_type() -> None:
    outward = CoordinateSystemTransform(
        from_convention=_convention("blender"), to_convention=_convention("ros")
    )
    homeward = CoordinateSystemTransform(
        from_convention=_convention("ros"), to_convention=_convention("blender")
    )
    displacement = Displacement.from_xyz(x=1.0, y=2.0, z=3.0)
    converted = outward.convert_displacement(displacement=displacement)
    assert isinstance(converted, Displacement)
    np.testing.assert_allclose(
        homeward.convert_displacement(displacement=converted).array,
        displacement.array,
        atol=1e-12,
    )


def test_a_convention_rejects_non_perpendicular_axes() -> None:
    with pytest.raises(ValueError, match="mutually perpendicular"):
        CoordinateSystemConvention(
            name="broken",
            description="",
            x_direction=AnatomicalDirection.RIGHT,
            y_direction=AnatomicalDirection.LEFT,
            z_direction=AnatomicalDirection.UP,
        )


def test_the_registry_rejects_a_declared_handedness_that_contradicts_the_axes() -> None:
    with pytest.raises(ValueError, match="contradicts its axes"):
        CoordinateSystemRegistry.from_document(
            document={
                "default": "mine",
                "conventions": {
                    "mine": {
                        "x_axis": "forward",
                        "y_axis": "right",
                        "z_axis": "up",
                        "handedness": "right",  # forward/right/up is actually left-handed
                    }
                },
            },
            source="<test>",
        )


def test_the_registry_rejects_a_missing_axis_key() -> None:
    with pytest.raises(ValueError, match="missing 'z_axis'"):
        CoordinateSystemRegistry.from_document(
            document={
                "default": "mine",
                "conventions": {"mine": {"x_axis": "right", "y_axis": "forward"}},
            },
            source="<test>",
        )


def test_the_registry_rejects_a_default_that_is_not_defined() -> None:
    with pytest.raises(ValueError, match="not one of the defined conventions"):
        CoordinateSystemRegistry.from_document(
            document={
                "default": "nope",
                "conventions": {
                    "blender": {"x_axis": "right", "y_axis": "forward", "z_axis": "up"}
                },
            },
            source="<test>",
        )


def test_the_registry_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown coordinate system"):
        _registry().get(name="diagonal")
