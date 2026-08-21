"""Tests for the landmark and rigid-body-segment definitions."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
)
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.point_ring_buffer import PointRingBuffer
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.rigid_body_segment import (
    RigidBodySegment,
    calculate_bases_for_segments,
)
from skellyforge.core.skeleton_parts.landmark_name_resolver import LandmarkNameResolver
from skellyforge.core.skeleton_parts.underspecified_rigid_body_segment import (
    UnderspecifiedRigidBodySegment,
)


def _landmark(
    *,
    name: str,
    position: tuple[float, float, float],
    segment: str = "upper_arm.L",
    aliases: tuple[str, ...] = (),
) -> AnatomicalLandmark:
    return AnatomicalLandmark(
        name=name,
        anatomical_definition=f"the {name}",
        local_position=Point.from_xyz(x=position[0], y=position[1], z=position[2]),
        segment=segment,
        aliases=aliases,
    )


def _segment(*, name: str = "upper_arm.L") -> RigidBodySegment:
    landmarks = {
        "shoulder": _landmark(name="shoulder", position=(0.0, 0.0, 0.0), segment=name),
        "elbow": _landmark(name="elbow", position=(0.0, 3.0, 0.0), segment=name),
        "lateral_epicondyle": _landmark(
            name="lateral_epicondyle", position=(1.0, 3.0, 0.0), segment=name
        ),
    }
    return RigidBodySegment(
        name=name,
        landmarks=landmarks,
        frame_definition=ReferenceFrameDefinition(
            origin_point_name="shoulder",
            primary_axis=SpatialAxis.Y,
            primary_point_name="elbow",
            secondary_axis=SpatialAxis.X,
            secondary_point_name="lateral_epicondyle",
        ),
    )


# ── AnatomicalLandmark ────────────────────────────────────────────────


def test_landmark_holds_a_single_rest_position() -> None:
    landmark = _landmark(name="elbow", position=(1.0, 2.0, 3.0))
    assert landmark.local_position.batch_shape == ()
    assert "elbow" in str(landmark)


def test_landmark_rejects_a_batched_rest_position() -> None:
    with pytest.raises(ValueError, match="single point"):
        AnatomicalLandmark(
            name="elbow",
            anatomical_definition="the elbow",
            local_position=Point.from_array(values=np.zeros(shape=(10, 3))),
            segment="upper_arm.L",
        )


@pytest.mark.parametrize(
    "name,definition,segment", [("", "d", "s"), ("n", "", "s"), ("n", "d", "")]
)
def test_landmark_rejects_empty_strings(name: str, definition: str, segment: str) -> None:
    with pytest.raises(ValueError):
        AnatomicalLandmark(
            name=name,
            anatomical_definition=definition,
            local_position=Point.from_xyz(x=0.0, y=0.0, z=0.0),
            segment=segment,
        )


# ── RigidBodySegment ──────────────────────────────────────────────────


def test_segment_length_comes_from_rest_positions() -> None:
    assert _segment().length == pytest.approx(3.0)


def test_segment_str_names_its_axes() -> None:
    assert "+y -> elbow" in str(_segment())


def test_segment_rejects_fewer_than_three_landmarks() -> None:
    landmarks = {
        "shoulder": _landmark(name="shoulder", position=(0.0, 0.0, 0.0)),
        "elbow": _landmark(name="elbow", position=(0.0, 3.0, 0.0)),
    }
    with pytest.raises(ValueError, match="at least 3 landmarks"):
        RigidBodySegment(
            name="upper_arm.L",
            landmarks=landmarks,
            frame_definition=ReferenceFrameDefinition(
                origin_point_name="shoulder",
                primary_axis=SpatialAxis.Y,
                primary_point_name="elbow",
                secondary_axis=SpatialAxis.X,
                secondary_point_name="missing",
            ),
        )


def test_segment_rejects_a_frame_definition_naming_absent_landmarks() -> None:
    segment = _segment()
    with pytest.raises(ValueError, match="does not have"):
        RigidBodySegment(
            name=segment.name,
            landmarks=segment.landmarks,
            frame_definition=ReferenceFrameDefinition(
                origin_point_name="shoulder",
                primary_axis=SpatialAxis.Y,
                primary_point_name="elbow",
                secondary_axis=SpatialAxis.X,
                secondary_point_name="nonexistent_landmark",
            ),
        )


def test_segment_rejects_landmarks_keyed_by_the_wrong_name() -> None:
    segment = _segment()
    mis_keyed = dict(segment.landmarks)
    mis_keyed["wrong_key"] = mis_keyed.pop("lateral_epicondyle")
    with pytest.raises(ValueError, match="keyed by their own name"):
        RigidBodySegment(
            name=segment.name, landmarks=mis_keyed, frame_definition=segment.frame_definition
        )


@pytest.mark.parametrize("name", ["Upper_Arm", "upper arm", "", "upperArm"])
def test_segment_rejects_non_snake_case_names(name: str) -> None:
    segment = _segment()
    with pytest.raises(ValueError, match="snake_case"):
        RigidBodySegment(
            name=name, landmarks=segment.landmarks, frame_definition=segment.frame_definition
        )


# ── UnderspecifiedRigidBodySegment ────────────────────────────────────


def _underspecified_segment() -> UnderspecifiedRigidBodySegment:
    return UnderspecifiedRigidBodySegment(
        name="forearm.L",
        origin_landmark=_landmark(name="elbow", position=(0.0, 0.0, 0.0), segment="forearm.L"),
        distal_landmark=_landmark(name="wrist", position=(0.0, -2.0, 0.0), segment="forearm.L"),
        primary_axis=SpatialAxis.NEGATIVE_Y,
    )


def test_underspecified_segment_has_no_basis_method() -> None:
    # The whole reason this type exists: two landmarks cannot determine roll, so there is
    # no calculate_basis to accidentally call.
    assert not hasattr(_underspecified_segment(), "calculate_basis")


def test_underspecified_segment_length_and_direction() -> None:
    segment = _underspecified_segment()
    assert segment.length == pytest.approx(2.0)
    points = {
        "elbow": Point.from_xyz(x=0.0, y=0.0, z=0.0),
        "wrist": Point.from_xyz(x=0.0, y=-5.0, z=0.0),
    }
    # The distal point lies on -y, and the axis is NEGATIVE_Y, so the direction is +y.
    np.testing.assert_allclose(
        segment.calculate_direction(points=points).array, [0.0, 1.0, 0.0], atol=1e-12
    )


def test_underspecified_segment_direction_is_vectorized_over_time() -> None:
    segment = _underspecified_segment()
    number_of_frames = 40
    points = {
        "elbow": Point.from_array(values=np.zeros(shape=(number_of_frames, 3))),
        "wrist": Point.from_array(
            values=np.tile([0.0, -1.0, 0.0], reps=(number_of_frames, 1))
        ),
    }
    assert segment.calculate_direction(points=points).batch_shape == (number_of_frames,)


def test_underspecified_segment_rejects_a_repeated_landmark() -> None:
    landmark = _landmark(name="elbow", position=(0.0, 0.0, 0.0), segment="forearm.L")
    with pytest.raises(ValueError, match="must differ"):
        UnderspecifiedRigidBodySegment(
            name="forearm.L",
            origin_landmark=landmark,
            distal_landmark=landmark,
            primary_axis=SpatialAxis.Y,
        )


# ── Batched multi-segment solve ───────────────────────────────────────


def _many_segments(*, count: int) -> tuple[list[RigidBodySegment], dict[str, Point]]:
    rng = np.random.default_rng(seed=5)
    segments: list[RigidBodySegment] = []
    points: dict[str, Point] = {}
    for index in range(count):
        names = [f"origin_{index}", f"primary_{index}", f"secondary_{index}"]
        segment_name = f"segment_{index}"
        landmarks = {
            name: _landmark(name=name, position=(0.0, 1.0, 2.0), segment=segment_name)
            for name in names
        }
        # Alternate conventions so the grouping logic is actually exercised.
        secondary_axis = SpatialAxis.Z if index % 2 == 0 else SpatialAxis.NEGATIVE_X
        segments.append(
            RigidBodySegment(
                name=segment_name,
                landmarks=landmarks,
                frame_definition=ReferenceFrameDefinition(
                    origin_point_name=names[0],
                    primary_axis=SpatialAxis.Y,
                    primary_point_name=names[1],
                    secondary_axis=secondary_axis,
                    secondary_point_name=names[2],
                ),
            )
        )
        for name in names:
            points[name] = Point.from_array(values=rng.normal(size=3))
    return segments, points


def test_batched_solve_matches_solving_each_segment_separately() -> None:
    segments, points = _many_segments(count=8)
    batched = calculate_bases_for_segments(segments=segments, points=points)
    for segment in segments:
        separate = calculate_orthonormal_basis(
            points=points, definition=segment.frame_definition
        )
        np.testing.assert_allclose(
            batched[segment.name].local_from_world_matrix,
            separate.local_from_world_matrix,
            atol=1e-12,
        )


def test_batched_solve_works_over_a_rolling_window() -> None:
    segments, _ = _many_segments(count=4)
    rng = np.random.default_rng(seed=6)
    landmark_names = [
        name for segment in segments for name in segment.frame_definition_landmark_names
    ]
    buffer = PointRingBuffer.for_point_names(point_names=landmark_names, capacity=25)
    for _ in range(40):
        buffer.append(positions=Point.from_array(values=rng.normal(size=(len(landmark_names), 3))))

    bases = calculate_bases_for_segments(segments=segments, points=buffer.window_by_name())
    assert set(bases) == {segment.name for segment in segments}
    for basis in bases.values():
        assert basis.x_axis.batch_shape == (25,)


def test_batched_solve_matches_the_single_frame_solve_over_a_window() -> None:
    segments, _ = _many_segments(count=3)
    rng = np.random.default_rng(seed=8)
    landmark_names = [
        name for segment in segments for name in segment.frame_definition_landmark_names
    ]
    buffer = PointRingBuffer.for_point_names(point_names=landmark_names, capacity=10)
    for _ in range(10):
        buffer.append(positions=Point.from_array(values=rng.normal(size=(len(landmark_names), 3))))

    windowed = calculate_bases_for_segments(segments=segments, points=buffer.window_by_name())
    newest = calculate_bases_for_segments(segments=segments, points=buffer.latest_by_name())
    for segment in segments:
        np.testing.assert_allclose(
            windowed[segment.name].local_from_world_matrix[-1],
            newest[segment.name].local_from_world_matrix,
            atol=1e-12,
        )


def test_batched_solve_rejects_duplicate_segment_names() -> None:
    segment = _segment()
    points = {
        name: Point.from_xyz(x=float(index), y=1.0, z=2.0)
        for index, name in enumerate(segment.frame_definition_landmark_names)
    }
    with pytest.raises(ValueError, match="unique"):
        calculate_bases_for_segments(segments=[segment, segment], points=points)


# ── Aliases ───────────────────────────────────────────────────────────


def test_all_names_puts_the_canonical_name_first() -> None:
    landmark = _landmark(
        name="shoulder.L", position=(0.0, 0.0, 0.0), aliases=("LSHO", "left_shoulder")
    )
    assert landmark.all_names == ("shoulder.L", "LSHO", "left_shoulder")
    assert "aka LSHO" in str(landmark)


def test_landmark_with_no_aliases_answers_only_to_its_name() -> None:
    landmark = _landmark(name="shoulder.L", position=(0.0, 0.0, 0.0))
    assert landmark.all_names == ("shoulder.L",)
    assert "aka" not in str(landmark)


@pytest.mark.parametrize(
    "aliases,message",
    [
        (("",), "non-empty"),
        (("LSHO", "LSHO"), "distinct"),
        (("shoulder.L",), "differ from the canonical name"),
    ],
    ids=["empty", "duplicated", "same-as-name"],
)
def test_landmark_rejects_bad_aliases(aliases: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _landmark(name="shoulder.L", position=(0.0, 0.0, 0.0), aliases=tuple(aliases))


def test_aliases_are_an_immutable_tuple() -> None:
    landmark = _landmark(name="shoulder.L", position=(0.0, 0.0, 0.0), aliases=("LSHO",))
    with pytest.raises(AttributeError):
        landmark.aliases.append("nope")


def test_segment_carries_aliases_too() -> None:
    segment = _segment()
    aliased = RigidBodySegment(
        name=segment.name,
        landmarks=segment.landmarks,
        frame_definition=segment.frame_definition,
        aliases=("humerus.L", "arm_upper.L"),
    )
    assert aliased.all_names == ("upper_arm.L", "humerus.L", "arm_upper.L")


def test_underspecified_segment_carries_aliases_too() -> None:
    segment = _underspecified_segment()
    aliased = UnderspecifiedRigidBodySegment(
        name=segment.name,
        origin_landmark=segment.origin_landmark,
        distal_landmark=segment.distal_landmark,
        primary_axis=segment.primary_axis,
        aliases=("radius_ulna.L",),
    )
    assert aliased.all_names == ("forearm.L", "radius_ulna.L")


def test_segment_rejects_landmarks_whose_names_collide_through_an_alias() -> None:
    # The elbow's alias is the shoulder's canonical name - a segment can see that, even
    # though neither landmark can see it alone.
    landmarks = {
        "shoulder": _landmark(name="shoulder", position=(0.0, 0.0, 0.0)),
        "elbow": _landmark(name="elbow", position=(0.0, 3.0, 0.0), aliases=("shoulder",)),
        "lateral_epicondyle": _landmark(name="lateral_epicondyle", position=(1.0, 3.0, 0.0)),
    }
    with pytest.raises(ValueError, match="claimed by both"):
        RigidBodySegment(
            name="upper_arm.L",
            landmarks=landmarks,
            frame_definition=_segment().frame_definition,
        )


# ── LandmarkNameResolver ──────────────────────────────────────────────


def _aliased_landmarks() -> list[AnatomicalLandmark]:
    return [
        _landmark(name="shoulder", position=(0.0, 0.0, 0.0), aliases=("LSHO",)),
        _landmark(name="elbow", position=(0.0, 3.0, 0.0), aliases=("LELB", "elbow_joint")),
        _landmark(name="lateral_epicondyle", position=(1.0, 3.0, 0.0)),
    ]


def test_resolver_maps_aliases_and_canonical_names_alike() -> None:
    resolver = LandmarkNameResolver.from_landmarks(landmarks=_aliased_landmarks())
    assert resolver.resolve(name="LSHO") == "shoulder"
    assert resolver.resolve(name="shoulder") == "shoulder"
    assert resolver.resolve(name="elbow_joint") == "elbow"


def test_resolver_rejects_unknown_names() -> None:
    resolver = LandmarkNameResolver.from_landmarks(landmarks=_aliased_landmarks())
    with pytest.raises(KeyError, match="Unknown landmark name"):
        resolver.resolve(name="RSHO")


def test_resolver_enforces_global_uniqueness() -> None:
    colliding = [
        _landmark(name="shoulder", position=(0.0, 0.0, 0.0), aliases=("JOINT",)),
        _landmark(name="elbow", position=(0.0, 3.0, 0.0), aliases=("JOINT",)),
    ]
    with pytest.raises(ValueError, match="globally unique"):
        LandmarkNameResolver.from_landmarks(landmarks=colliding)


def test_resolver_catches_an_alias_shadowing_another_landmarks_name() -> None:
    colliding = [
        _landmark(name="shoulder", position=(0.0, 0.0, 0.0)),
        _landmark(name="elbow", position=(0.0, 3.0, 0.0), aliases=("shoulder",)),
    ]
    with pytest.raises(ValueError, match="globally unique"):
        LandmarkNameResolver.from_landmarks(landmarks=colliding)


def test_resolver_rejects_an_empty_landmark_set() -> None:
    with pytest.raises(ValueError, match="no landmarks"):
        LandmarkNameResolver.from_landmarks(landmarks=[])


def test_resolve_points_rekeys_a_mapping_canonically() -> None:
    resolver = LandmarkNameResolver.from_landmarks(landmarks=_aliased_landmarks())
    observed = {
        "LSHO": Point.from_xyz(x=1.0, y=0.0, z=0.0),
        "elbow_joint": Point.from_xyz(x=0.0, y=1.0, z=0.0),
    }
    resolved = resolver.resolve_points(points=observed)
    assert sorted(resolved) == ["elbow", "shoulder"]
    np.testing.assert_allclose(resolved["shoulder"].array, [1.0, 0.0, 0.0])


def test_resolve_points_rejects_two_keys_for_the_same_landmark() -> None:
    resolver = LandmarkNameResolver.from_landmarks(landmarks=_aliased_landmarks())
    with pytest.raises(ValueError, match="resolve to the landmark"):
        resolver.resolve_points(
            points={
                "LSHO": Point.from_xyz(x=1.0, y=0.0, z=0.0),
                "shoulder": Point.from_xyz(x=2.0, y=0.0, z=0.0),
            }
        )


def test_aliased_stream_solves_a_segment_end_to_end() -> None:
    # The whole point of aliases: data arriving under foreign names resolves once at the
    # boundary, and the solver only ever sees canonical names.
    landmarks = {landmark.name: landmark for landmark in _aliased_landmarks()}
    segment = RigidBodySegment(
        name="upper_arm.L",
        landmarks=landmarks,
        frame_definition=ReferenceFrameDefinition(
            origin_point_name="shoulder",
            primary_axis=SpatialAxis.Y,
            primary_point_name="elbow",
            secondary_axis=SpatialAxis.X,
            secondary_point_name="lateral_epicondyle",
        ),
    )
    resolver = LandmarkNameResolver.from_landmarks(landmarks=landmarks.values())
    foreign_stream = {
        "LSHO": Point.from_xyz(x=0.0, y=0.0, z=0.0),
        "LELB": Point.from_xyz(x=0.0, y=3.0, z=0.0),
        "lateral_epicondyle": Point.from_xyz(x=1.0, y=3.0, z=0.0),
    }
    bases = calculate_bases_for_segments(
        segments=[segment], points=resolver.resolve_points(points=foreign_stream)
    )
    np.testing.assert_allclose(
        bases["upper_arm.L"].y_axis.array, [0.0, 1.0, 0.0], atol=1e-12
    )


def test_resolve_all_keys_a_ring_buffer_canonically() -> None:
    # Resolution happens ONCE here, at buffer construction - never per frame.
    resolver = LandmarkNameResolver.from_landmarks(landmarks=_aliased_landmarks())
    buffer = PointRingBuffer(
        point_names=resolver.resolve_all(names=["LSHO", "LELB", "lateral_epicondyle"]),
        capacity=4,
    )
    assert buffer.point_names == ("shoulder", "elbow", "lateral_epicondyle")
