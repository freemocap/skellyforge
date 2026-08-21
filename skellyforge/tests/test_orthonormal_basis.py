"""Tests for the geometry module: spatial vector algebra, transforms, and the
Gram-Schmidt orthonormal basis construction built on top of them."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
)
from skellyforge.core.math.geometry.orthonormal_basis.handedness import (
    Handedness,
    LeftHandedCoordinateSystemWarning,
)
from skellyforge.core.math.geometry.orthonormal_basis.orthonormal_basis import OrthonormalBasis
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import (
    Displacement,
    Point,
    UnitVector,
    Vector3Array,
)
from skellyforge.core.math.geometry.transform_math import Transform

SIMPLE_POINTS: dict[str, Point] = {
    "origin": Point.from_xyz(x=0.0, y=0.0, z=0.0),
    "front": Point.from_xyz(x=2.0, y=0.0, z=0.0),
    "left": Point.from_xyz(x=1.0, y=3.0, z=0.0),
}

SCATTERED_POINTS: dict[str, Point] = {
    "origin": Point.from_xyz(x=1.0, y=-2.0, z=0.5),
    "primary": Point.from_xyz(x=1.7, y=0.3, z=2.5),
    "secondary": Point.from_xyz(x=-0.4, y=1.1, z=0.9),
}

SIGNED_AXIS_PAIRS: list[tuple[SpatialAxis, SpatialAxis]] = [
    (primary, secondary)
    for primary in SpatialAxis
    for secondary in SpatialAxis
    if primary.index != secondary.index
]


def _random_point_dict(*, seed: int, shape: tuple[int, ...] = (3,)) -> dict[str, Point]:
    rng = np.random.default_rng(seed=seed)
    return {
        "origin": Point.from_array(values=rng.normal(size=shape)),
        "primary": Point.from_array(values=rng.normal(size=shape)),
        "secondary": Point.from_array(values=rng.normal(size=shape)),
    }


# ── Spatial vector algebra ────────────────────────────────────────────


def test_point_minus_point_is_a_displacement() -> None:
    difference = Point.from_xyz(x=3.0, y=0.0, z=0.0) - Point.from_xyz(x=1.0, y=0.0, z=0.0)
    assert isinstance(difference, Displacement)
    np.testing.assert_allclose(difference.array, [2.0, 0.0, 0.0])


def test_point_plus_displacement_is_a_point() -> None:
    moved = Point.from_xyz(x=1.0, y=1.0, z=1.0) + Displacement.from_xyz(x=1.0, y=0.0, z=0.0)
    assert isinstance(moved, Point)
    np.testing.assert_allclose(moved.array, [2.0, 1.0, 1.0])


def test_point_minus_displacement_is_a_point() -> None:
    moved = Point.from_xyz(x=1.0, y=1.0, z=1.0) - Displacement.from_xyz(x=1.0, y=0.0, z=0.0)
    assert isinstance(moved, Point)
    np.testing.assert_allclose(moved.array, [0.0, 1.0, 1.0])


def test_adding_two_points_is_rejected() -> None:
    # beartype rejects it on the type hint; the isinstance guard behind it raises
    # TypeError on its own wherever beartype is not active.
    with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
        Point.from_xyz(x=1.0, y=0.0, z=0.0) + Point.from_xyz(x=1.0, y=0.0, z=0.0)


def test_adding_a_point_to_a_displacement_is_rejected() -> None:
    with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
        Displacement.from_xyz(x=1.0, y=0.0, z=0.0) + Point.from_xyz(x=1.0, y=0.0, z=0.0)


def test_displacements_add_and_negate() -> None:
    first = Displacement.from_xyz(x=1.0, y=0.0, z=0.0)
    second = Displacement.from_xyz(x=0.0, y=2.0, z=0.0)
    np.testing.assert_allclose((first + second).array, [1.0, 2.0, 0.0])
    np.testing.assert_allclose((first - second).array, [1.0, -2.0, 0.0])
    np.testing.assert_allclose((-first).array, [-1.0, 0.0, 0.0])
    assert isinstance(first + second, Displacement)


def test_normalizing_gives_a_unit_vector() -> None:
    direction = Displacement.from_xyz(x=0.0, y=5.0, z=0.0).normalized(description="test")
    assert isinstance(direction, UnitVector)
    np.testing.assert_allclose(direction.array, [0.0, 1.0, 0.0])


def test_normalizing_a_zero_displacement_raises() -> None:
    with pytest.raises(ValueError, match="coincident"):
        Displacement.from_xyz(x=0.0, y=0.0, z=0.0).normalized(description="a test vector")


def test_unit_vector_rejects_non_unit_input() -> None:
    with pytest.raises(ValueError, match="unit length"):
        UnitVector.from_xyz(x=2.0, y=0.0, z=0.0)


def test_unit_vector_negation_stays_unit() -> None:
    direction = UnitVector.from_xyz(x=1.0, y=0.0, z=0.0)
    assert isinstance(-direction, UnitVector)
    np.testing.assert_allclose((-direction).array, [-1.0, 0.0, 0.0])


def test_cross_of_perpendicular_unit_vectors() -> None:
    x_direction = UnitVector.from_xyz(x=1.0, y=0.0, z=0.0)
    y_direction = UnitVector.from_xyz(x=0.0, y=1.0, z=0.0)
    np.testing.assert_allclose(x_direction.cross(other=y_direction).array, [0.0, 0.0, 1.0])


def test_cross_of_non_perpendicular_unit_vectors_raises() -> None:
    # The cross product's length is the sine of the angle between them, so it is only a
    # unit vector when they are perpendicular.
    x_direction = UnitVector.from_xyz(x=1.0, y=0.0, z=0.0)
    skewed = Displacement.from_xyz(x=1.0, y=1.0, z=0.0).normalized(description="skewed")
    with pytest.raises(ValueError, match="unit length"):
        x_direction.cross(other=skewed)


def test_scaling_a_unit_vector_gives_a_displacement() -> None:
    scaled = UnitVector.from_xyz(x=1.0, y=0.0, z=0.0).scaled_by(factors=3.0)
    assert isinstance(scaled, Displacement)
    np.testing.assert_allclose(scaled.array, [3.0, 0.0, 0.0])


def test_scaling_batched_vectors_uses_one_factor_per_batch_element() -> None:
    displacement = Displacement.from_array(values=np.ones(shape=(4, 3)))
    scaled = displacement.scaled_by(factors=np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(scaled.array[:, 0], [1.0, 2.0, 3.0, 4.0])


def test_vectors_are_immutable_after_construction() -> None:
    point = Point.from_xyz(x=1.0, y=2.0, z=3.0)
    with pytest.raises(Exception):
        point.array = np.zeros(3)


def test_bad_trailing_dimension_raises() -> None:
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        Point.from_array(values=np.zeros(2))


def test_non_finite_values_raise() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        Point.from_array(values=[np.nan, 0.0, 0.0])


def test_component_accessors_work_at_every_batch_rank() -> None:
    single = Point.from_xyz(x=1.0, y=2.0, z=3.0)
    assert (float(single.x), float(single.y), float(single.z)) == (1.0, 2.0, 3.0)
    assert single.batch_shape == ()
    batched = Point.from_array(values=np.zeros(shape=(7, 5, 3)))
    assert batched.batch_shape == (7, 5)
    assert batched.x.shape == (7, 5)


def test_prevalidated_construction_skips_the_scans() -> None:
    # The trusted path is what keeps derived results and buffer views O(1) instead of
    # O(batch). It asserts rather than checks, so it will happily wrap nonsense - which is
    # exactly why only internally-derived data may use it.
    nonsense = UnitVector.from_prevalidated_array(array=np.array([5.0, 5.0, 5.0]))
    assert float(nonsense.x) == 5.0
    with pytest.raises(ValueError, match="unit length"):
        UnitVector.from_array(values=[5.0, 5.0, 5.0])


def test_derived_results_stay_correct_through_the_trusted_path() -> None:
    rng = np.random.default_rng(seed=21)
    first = Point.from_array(values=rng.normal(size=(100, 3)))
    second = Point.from_array(values=rng.normal(size=(100, 3)))
    direction = (second - first).normalized(description="test displacement")
    # `norm()` lives on Displacement - a UnitVector's length is 1 by construction.
    np.testing.assert_allclose(
        direction.as_displacement().norm(), np.ones(100), atol=1e-12
    )


def test_basis_batch_indexing_matches_a_directly_solved_frame() -> None:
    rng = np.random.default_rng(seed=22)
    number_of_frames = 30
    points = {
        name: Point.from_array(values=rng.normal(size=(number_of_frames, 3)))
        for name in ("origin", "primary", "secondary")
    }
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="primary",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="secondary",
    )
    batched = calculate_orthonormal_basis(points=points, definition=definition)
    single = calculate_orthonormal_basis(
        points={name: Point(array=point.array[7]) for name, point in points.items()},
        definition=definition,
    )
    np.testing.assert_allclose(
        batched.at_batch_index(index=7).local_from_world_matrix,
        single.local_from_world_matrix,
        atol=1e-12,
    )


def test_the_three_vector_types_are_not_interchangeable() -> None:
    assert issubclass(Point, Vector3Array)
    assert not issubclass(Point, Displacement)
    assert not issubclass(Displacement, UnitVector)


# ── RotationQuaternion immutability ───────────────────────────────────


def test_quaternion_is_frozen_and_unit_on_construction() -> None:
    quaternion = RotationQuaternion.from_components(w=1.0, x=2.0, y=3.0, z=4.0)
    assert quaternion.w**2 + quaternion.x**2 + quaternion.y**2 + quaternion.z**2 == (
        pytest.approx(1.0)
    )
    with pytest.raises(Exception):
        quaternion.w = 0.5


def test_quaternion_constructor_rejects_unnormalized_components() -> None:
    with pytest.raises(ValueError, match="unit length"):
        RotationQuaternion(w=1.0, x=2.0, y=3.0, z=4.0)


def test_quaternion_factory_rejects_a_near_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="near-zero"):
        RotationQuaternion.from_components(w=0.0, x=0.0, y=0.0, z=0.0)


def test_quaternion_products_stay_unit() -> None:
    first = RotationQuaternion.from_rotation_vector(np.array([0.3, -1.2, 0.7]))
    second = RotationQuaternion.from_rotation_vector(np.array([2.0, 0.1, -0.4]))
    product = first * second
    assert product.dot(product) == pytest.approx(1.0)


# ── Transform ─────────────────────────────────────────────────────────


def test_identity_transform_leaves_points_alone() -> None:
    points = Point.from_array(values=np.random.default_rng(seed=3).normal(size=(10, 3)))
    np.testing.assert_allclose(Transform.identity().apply(points=points).array, points.array)


def test_transform_inverse_round_trips() -> None:
    transform = Transform(
        rotation=RotationQuaternion.from_rotation_vector(np.array([0.4, 0.2, -1.1])),
        translation=Displacement.from_xyz(x=1.0, y=2.0, z=3.0),
    )
    points = Point.from_array(values=np.random.default_rng(seed=4).normal(size=(10, 3)))
    round_tripped = transform.inverse().apply(points=transform.apply(points=points))
    np.testing.assert_allclose(round_tripped.array, points.array, atol=1e-12)


def test_transform_rejects_a_batched_translation() -> None:
    with pytest.raises(ValueError, match="single translation"):
        Transform(
            rotation=RotationQuaternion.identity(),
            translation=Displacement.from_array(values=np.zeros(shape=(5, 3))),
        )


# ── Orthonormal basis construction ────────────────────────────────────


def test_primary_axis_points_exactly_at_primary_point() -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="front",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="left",
    )
    basis = calculate_orthonormal_basis(points=SIMPLE_POINTS, definition=definition)
    np.testing.assert_allclose(basis.x_axis.array, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.y_axis.array, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.z_axis.array, [0.0, 0.0, 1.0], atol=1e-12)


def test_negative_primary_axis_flips_that_basis_vector() -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.NEGATIVE_X,
        primary_point_name="front",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="left",
    )
    basis = calculate_orthonormal_basis(points=SIMPLE_POINTS, definition=definition)
    np.testing.assert_allclose(basis.x_axis.array, [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.y_axis.array, [0.0, 1.0, 0.0], atol=1e-12)
    # Right-handedness is preserved by flipping the tertiary axis too.
    np.testing.assert_allclose(basis.z_axis.array, [0.0, 0.0, -1.0], atol=1e-12)


def test_negative_secondary_axis_flips_that_basis_vector() -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="front",
        secondary_axis=SpatialAxis.NEGATIVE_Y,
        secondary_point_name="left",
    )
    basis = calculate_orthonormal_basis(points=SIMPLE_POINTS, definition=definition)
    np.testing.assert_allclose(basis.x_axis.array, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.y_axis.array, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.z_axis.array, [0.0, 0.0, -1.0], atol=1e-12)


def test_axis_along_returns_the_signed_direction() -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="front",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="left",
    )
    basis = calculate_orthonormal_basis(points=SIMPLE_POINTS, definition=definition)
    np.testing.assert_allclose(basis.axis_along(axis=SpatialAxis.Y).array, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        basis.axis_along(axis=SpatialAxis.NEGATIVE_Y).array, [0.0, -1.0, 0.0]
    )


def test_named_points_land_on_their_declared_signed_axes() -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.NEGATIVE_Z,
        primary_point_name="primary",
        secondary_axis=SpatialAxis.NEGATIVE_X,
        secondary_point_name="secondary",
    )
    basis = calculate_orthonormal_basis(points=SCATTERED_POINTS, definition=definition)
    local = basis.transform_named_points_to_local(points=SCATTERED_POINTS)
    np.testing.assert_allclose(local["origin"].array, np.zeros(3), atol=1e-12)
    # The primary point sits purely on -z.
    assert float(local["primary"].z) < 0.0
    np.testing.assert_allclose(local["primary"].array[:2], 0.0, atol=1e-12)
    # The secondary point has negative x and zero y after orthogonalization.
    assert float(local["secondary"].x) < 0.0
    assert float(local["secondary"].y) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("primary_axis,secondary_axis", SIGNED_AXIS_PAIRS)
def test_every_signed_axis_pairing_puts_its_points_on_its_declared_axes(
    primary_axis: SpatialAxis, secondary_axis: SpatialAxis
) -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=primary_axis,
        primary_point_name="primary",
        secondary_axis=secondary_axis,
        secondary_point_name="secondary",
    )
    basis = calculate_orthonormal_basis(points=SCATTERED_POINTS, definition=definition)
    local = basis.transform_named_points_to_local(points=SCATTERED_POINTS)

    # The primary point lies exactly on the declared half-axis, and nowhere else.
    primary_local = local["primary"].array
    assert np.sign(primary_local[primary_axis.index]) == primary_axis.sign
    off_axis = [
        value for index, value in enumerate(primary_local) if index != primary_axis.index
    ]
    np.testing.assert_allclose(off_axis, 0.0, atol=1e-12)

    # The secondary point lies on the declared side of its axis, with no tertiary component.
    secondary_local = local["secondary"].array
    assert np.sign(secondary_local[secondary_axis.index]) == secondary_axis.sign
    assert secondary_local[definition.tertiary_axis.index] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("primary_axis,secondary_axis", SIGNED_AXIS_PAIRS)
def test_every_signed_axis_pairing_is_orthonormal_and_right_handed(
    primary_axis: SpatialAxis, secondary_axis: SpatialAxis
) -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=primary_axis,
        primary_point_name="primary",
        secondary_axis=secondary_axis,
        secondary_point_name="secondary",
    )
    basis = calculate_orthonormal_basis(points=SCATTERED_POINTS, definition=definition)
    assert np.linalg.det(basis.local_from_world_matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.cross(basis.x_axis.array, basis.y_axis.array), basis.z_axis.array, atol=1e-12
    )


@pytest.mark.parametrize("primary_axis,secondary_axis", SIGNED_AXIS_PAIRS)
def test_every_signed_axis_pairing_can_be_built_left_handed(
    primary_axis: SpatialAxis, secondary_axis: SpatialAxis
) -> None:
    with pytest.warns(LeftHandedCoordinateSystemWarning):
        definition = ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=primary_axis,
            primary_point_name="primary",
            secondary_axis=secondary_axis,
            secondary_point_name="secondary",
            handedness=Handedness.LEFT_HANDED,
        )
    basis = calculate_orthonormal_basis(points=SCATTERED_POINTS, definition=definition)
    assert np.linalg.det(basis.local_from_world_matrix) == pytest.approx(-1.0)
    np.testing.assert_allclose(
        np.cross(basis.x_axis.array, basis.y_axis.array), -basis.z_axis.array, atol=1e-12
    )


def test_handedness_defaults_to_right_handed() -> None:
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="front",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="left",
    )
    assert definition.handedness is Handedness.RIGHT_HANDED


def test_left_handed_frame_only_flips_the_tertiary_axis() -> None:
    right_handed_definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="primary",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="secondary",
    )
    with pytest.warns(LeftHandedCoordinateSystemWarning):
        left_handed_definition = ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=SpatialAxis.X,
            primary_point_name="primary",
            secondary_axis=SpatialAxis.Y,
            secondary_point_name="secondary",
            handedness=Handedness.LEFT_HANDED,
        )
    right_handed = calculate_orthonormal_basis(
        points=SCATTERED_POINTS, definition=right_handed_definition
    )
    left_handed = calculate_orthonormal_basis(
        points=SCATTERED_POINTS, definition=left_handed_definition
    )
    np.testing.assert_allclose(left_handed.x_axis.array, right_handed.x_axis.array, atol=1e-12)
    np.testing.assert_allclose(left_handed.y_axis.array, right_handed.y_axis.array, atol=1e-12)
    np.testing.assert_allclose(left_handed.z_axis.array, -right_handed.z_axis.array, atol=1e-12)


def test_right_handed_frame_emits_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=SpatialAxis.X,
            primary_point_name="front",
            secondary_axis=SpatialAxis.NEGATIVE_Z,
            secondary_point_name="left",
        )


@pytest.mark.parametrize("handedness", list(Handedness))
def test_round_trip_transform_is_the_identity(handedness: Handedness) -> None:
    points = _random_point_dict(seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LeftHandedCoordinateSystemWarning)
        definition = ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=SpatialAxis.NEGATIVE_Y,
            primary_point_name="primary",
            secondary_axis=SpatialAxis.Z,
            secondary_point_name="secondary",
            handedness=handedness,
        )
    basis = calculate_orthonormal_basis(points=points, definition=definition)
    world_points = Point.from_array(
        values=np.random.default_rng(seed=11).normal(size=(50, 3))
    )
    local_points = basis.transform_points_to_local(points=world_points)
    np.testing.assert_allclose(
        basis.transform_points_to_world(points=local_points).array,
        world_points.array,
        atol=1e-12,
    )


def test_transform_preserves_pairwise_distances() -> None:
    points = _random_point_dict(seed=7)
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="primary",
        secondary_axis=SpatialAxis.NEGATIVE_Z,
        secondary_point_name="secondary",
    )
    basis = calculate_orthonormal_basis(points=points, definition=definition)
    world = np.random.default_rng(seed=12).normal(size=(20, 3))
    local = basis.transform_points_to_local(points=Point.from_array(values=world)).array
    world_distances = np.linalg.norm(world[:, None, :] - world[None, :, :], axis=-1)
    local_distances = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
    np.testing.assert_allclose(local_distances, world_distances, atol=1e-12)


def test_batched_trajectories_are_handled_frame_by_frame() -> None:
    rng = np.random.default_rng(seed=1234)
    number_of_frames = 500
    points: dict[str, Point] = {
        "origin": Point.from_array(values=rng.normal(size=(number_of_frames, 3))),
        "primary": Point.from_array(
            values=rng.normal(size=(number_of_frames, 3)) + np.array([5.0, 0.0, 0.0])
        ),
        "secondary": Point.from_array(
            values=rng.normal(size=(number_of_frames, 3)) + np.array([0.0, 5.0, 0.0])
        ),
    }
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.NEGATIVE_X,
        primary_point_name="primary",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="secondary",
    )
    basis = calculate_orthonormal_basis(points=points, definition=definition)
    assert basis.x_axis.batch_shape == (number_of_frames,)
    assert basis.local_from_world_matrix.shape == (number_of_frames, 3, 3)

    single_frame_basis = calculate_orthonormal_basis(
        points={
            name: Point(array=point.array[17]) for name, point in points.items()
        },
        definition=definition,
    )
    np.testing.assert_allclose(
        basis.local_from_world_matrix[17],
        single_frame_basis.local_from_world_matrix,
        atol=1e-12,
    )


def test_batched_named_point_transform_matches_the_batch_shape() -> None:
    rng = np.random.default_rng(seed=99)
    number_of_frames = 20
    points: dict[str, Point] = {
        "origin": Point.from_array(values=rng.normal(size=(number_of_frames, 3))),
        "primary": Point.from_array(
            values=rng.normal(size=(number_of_frames, 3)) + np.array([5.0, 0.0, 0.0])
        ),
        "secondary": Point.from_array(
            values=rng.normal(size=(number_of_frames, 3)) + np.array([0.0, 5.0, 0.0])
        ),
    }
    definition = ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="primary",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="secondary",
    )
    basis = calculate_orthonormal_basis(points=points, definition=definition)
    local = basis.transform_named_points_to_local(points=points)
    assert local["primary"].batch_shape == (number_of_frames,)
    np.testing.assert_allclose(local["origin"].array, 0.0, atol=1e-12)
    np.testing.assert_allclose(local["primary"].array[:, 1:], 0.0, atol=1e-12)


# ── Definition validation ─────────────────────────────────────────────


def test_same_cartesian_axis_for_both_definitions_is_rejected() -> None:
    with pytest.raises(ValueError, match="different cartesian axes"):
        ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=SpatialAxis.X,
            primary_point_name="front",
            secondary_axis=SpatialAxis.NEGATIVE_X,
            secondary_point_name="left",
        )


def test_repeated_point_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="points must differ"):
        ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=SpatialAxis.X,
            primary_point_name="front",
            secondary_axis=SpatialAxis.Y,
            secondary_point_name="front",
        )
    with pytest.raises(ValueError, match="must differ from"):
        ReferenceFrameDefinition(
            origin_point_name="front",
            primary_axis=SpatialAxis.X,
            primary_point_name="front",
            secondary_axis=SpatialAxis.Y,
            secondary_point_name="left",
        )


def _degenerate_definition() -> ReferenceFrameDefinition:
    return ReferenceFrameDefinition(
        origin_point_name="origin",
        primary_axis=SpatialAxis.X,
        primary_point_name="a",
        secondary_axis=SpatialAxis.Y,
        secondary_point_name="b",
    )


def test_missing_point_raises_key_error() -> None:
    with pytest.raises(KeyError, match="'a'"):
        calculate_orthonormal_basis(points=SIMPLE_POINTS, definition=_degenerate_definition())


@pytest.mark.parametrize(
    "b_values", [[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0]], ids=["parallel", "antiparallel"]
)
def test_collinear_defining_points_raise(b_values: list[float]) -> None:
    with pytest.raises(ValueError, match="collinear"):
        calculate_orthonormal_basis(
            points={
                "origin": Point.from_xyz(x=0.0, y=0.0, z=0.0),
                "a": Point.from_xyz(x=1.0, y=0.0, z=0.0),
                "b": Point.from_array(values=b_values),
            },
            definition=_degenerate_definition(),
        )


def test_coincident_points_raise() -> None:
    with pytest.raises(ValueError, match="coincident"):
        calculate_orthonormal_basis(
            points={
                "origin": Point.from_xyz(x=0.0, y=0.0, z=0.0),
                "a": Point.from_xyz(x=0.0, y=0.0, z=0.0),
                "b": Point.from_xyz(x=0.0, y=1.0, z=0.0),
            },
            definition=_degenerate_definition(),
        )


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_orthonormal_basis(
            points={
                "origin": Point.from_xyz(x=0.0, y=0.0, z=0.0),
                "a": Point.from_array(values=np.ones(shape=(10, 3))),
                "b": Point.from_xyz(x=0.0, y=1.0, z=0.0),
            },
            definition=_degenerate_definition(),
        )


# ── Basis invariants are enforced at construction ─────────────────────


def test_basis_construction_rejects_non_orthogonal_axes() -> None:
    skewed = Displacement.from_xyz(x=1.0, y=1.0, z=0.0).normalized(description="skewed")
    with pytest.raises(ValueError, match="not mutually orthogonal"):
        OrthonormalBasis(
            origin=Point.from_xyz(x=0.0, y=0.0, z=0.0),
            x_axis=UnitVector.from_xyz(x=1.0, y=0.0, z=0.0),
            y_axis=skewed,
            z_axis=UnitVector.from_xyz(x=0.0, y=0.0, z=1.0),
            handedness=Handedness.RIGHT_HANDED,
        )


@pytest.mark.parametrize(
    "handedness,z_sign",
    [(Handedness.RIGHT_HANDED, -1.0), (Handedness.LEFT_HANDED, 1.0)],
)
def test_basis_construction_rejects_mismatched_handedness(
    handedness: Handedness, z_sign: float
) -> None:
    with pytest.raises(ValueError, match="handedness does not match"):
        OrthonormalBasis(
            origin=Point.from_xyz(x=0.0, y=0.0, z=0.0),
            x_axis=UnitVector.from_xyz(x=1.0, y=0.0, z=0.0),
            y_axis=UnitVector.from_xyz(x=0.0, y=1.0, z=0.0),
            z_axis=UnitVector.from_xyz(x=0.0, y=0.0, z=z_sign),
            handedness=handedness,
        )


# ── Axis arithmetic ───────────────────────────────────────────────────


def test_signed_axis_index_sign_and_string_representation() -> None:
    assert str(SpatialAxis.NEGATIVE_Y) == "-y"
    assert str(SpatialAxis.Z) == "+z"
    assert SpatialAxis.NEGATIVE_Y.index == SpatialAxis.Y.index
    assert SpatialAxis.NEGATIVE_Y is not SpatialAxis.Y
    assert SpatialAxis.NEGATIVE_Y.sign == -1
    assert SpatialAxis.Y.sign == 1


def test_cyclic_sign_and_remaining_axis() -> None:
    assert SpatialAxis.X.cyclic_sign_toward(SpatialAxis.Y) == 1
    assert SpatialAxis.Y.cyclic_sign_toward(SpatialAxis.Z) == 1
    assert SpatialAxis.Z.cyclic_sign_toward(SpatialAxis.X) == 1
    assert SpatialAxis.Y.cyclic_sign_toward(SpatialAxis.X) == -1
    # The signed direction never changes the axis arithmetic, only the vectors do.
    assert SpatialAxis.NEGATIVE_X.cyclic_sign_toward(SpatialAxis.NEGATIVE_Y) == 1
    assert SpatialAxis.X.remaining_axis(SpatialAxis.Z) is SpatialAxis.Y
    assert SpatialAxis.NEGATIVE_X.remaining_axis(SpatialAxis.NEGATIVE_Z) is SpatialAxis.Y
    with pytest.raises(ValueError, match=r"both axes are \+x"):
        SpatialAxis.X.remaining_axis(SpatialAxis.X)
    # Opposite directions along one axis are still the same axis, and still rejected.
    with pytest.raises(ValueError, match=r"both axes are \+x"):
        SpatialAxis.X.cyclic_sign_toward(SpatialAxis.NEGATIVE_X)
