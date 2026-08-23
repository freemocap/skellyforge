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
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point


def _registry() -> CoordinateSystemRegistry:
    return CoordinateSystemRegistry.from_default_yaml()


def _convention(name: str) -> CoordinateSystemConvention:
    return _registry().get(name=name)


def _random_points(*, seed: int) -> Point:
    return Point.from_array(values=np.random.default_rng(seed).normal(size=(10, 3)))


def test_the_shipped_registry_defaults_to_blender() -> None:
    registry = _registry()
    assert registry.default_name == "blender"
    assert set(registry.conventions) == {"blender", "vrm", "ros", "isb", "unreal", "unity"}
    assert registry.default.handedness is Handedness.RIGHT_HANDED


@pytest.mark.parametrize(
    "name,expected_handedness",
    [
        ("blender", Handedness.RIGHT_HANDED),
        ("vrm", Handedness.RIGHT_HANDED),
        ("ros", Handedness.RIGHT_HANDED),
        ("isb", Handedness.RIGHT_HANDED),
        ("unreal", Handedness.LEFT_HANDED),
        ("unity", Handedness.LEFT_HANDED),
    ],
)
def test_every_shipped_convention_has_the_expected_handedness(
    name: str, expected_handedness: Handedness
) -> None:
    assert _convention(name).handedness is expected_handedness


def test_anatomical_direction_unit_vectors() -> None:
    np.testing.assert_allclose(AnatomicalDirection.RIGHT.unit_vector, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(AnatomicalDirection.FORWARD.unit_vector, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(AnatomicalDirection.UP.unit_vector, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(AnatomicalDirection.LEFT.unit_vector, [-1.0, 0.0, 0.0])


def test_parse_anatomical_direction_is_case_insensitive_and_rejects_unknown() -> None:
    assert parse_anatomical_direction(label="RIGHT") is AnatomicalDirection.RIGHT
    with pytest.raises(ValueError, match="unknown anatomical direction"):
        parse_anatomical_direction(label="diagonal")


def test_converting_blender_axes_into_vrm() -> None:
    transform = CoordinateSystemTransform(
        from_convention=_convention("blender"), to_convention=_convention("vrm")
    )
    # up in Blender is +Z; up in VRM is +Y.
    np.testing.assert_allclose(
        transform.convert_point(point=Point.from_xyz(x=0.0, y=0.0, z=1.0)).array,
        [0.0, 1.0, 0.0],
    )
    # forward in Blender is +Y; forward in VRM is +Z.
    np.testing.assert_allclose(
        transform.convert_point(point=Point.from_xyz(x=0.0, y=1.0, z=0.0)).array,
        [0.0, 0.0, 1.0],
    )
    # right in Blender is +X; right in VRM is -X (VRM's +X is left).
    np.testing.assert_allclose(
        transform.convert_point(point=Point.from_xyz(x=1.0, y=0.0, z=0.0)).array,
        [-1.0, 0.0, 0.0],
    )


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
