"""Round-trip synthetic ground-truth tests for skeleton hydration.

Generate a pose with the forward model (`build_rest_pose`), project it to landmark
positions, corrupt the landmarks, and solve back with `hydrate_skeleton`. The recovered
orientations must match the ones that were synthesized - exactly when there is no noise,
and within a noise-scaled tolerance when there is.

The tree the synthesis walks comes from `RestPose`, not from a second reading of the YAML.
A test that parsed the rest pose itself would be exercising its own parser rather than the
one that ships.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.rest_pose import RestPose, build_rest_pose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SkeletonPose
from skellyforge.type_overloads import FloatArray

SKELETON_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)
REST_POSE_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "rest_pose.yaml"
)

SHORT_SEGMENT_LENGTH_MILLIMETRES: float = 50.0


def _synthesize_and_hydrate(
    *, noise_scale: float, seed: int
) -> tuple[SkeletonDefinition, dict[str, RotationQuaternion], SkeletonPose]:
    """Perturb every joint, project to landmarks, add noise, and solve back."""
    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)

    generator = np.random.default_rng(seed)
    synthesized = {
        name: RotationQuaternion.from_rotation_vector(
            rotation_vector=generator.normal(scale=0.3, size=3)
        )
        * orientation
        for name, orientation in rest_pose.relative_orientations.items()
    }

    world_orientations, _world_origins, true_landmarks = build_rest_pose(
        skeleton=skeleton,
        parents=rest_pose.parents,
        connect_ats=rest_pose.connect_ats,
        orientations=synthesized,
    )

    observed = {
        name: Point.from_prevalidated_array(
            array=point.array + generator.normal(scale=noise_scale, size=3)
        )
        for name, point in true_landmarks.items()
    }

    return skeleton, world_orientations, hydrate_skeleton(
        skeleton=skeleton, observed=observed
    )


def _direction_error(
    *,
    recovered: RotationQuaternion,
    synthesized: RotationQuaternion,
    primary: FloatArray,
) -> float:
    """The angle in radians between the recovered and synthesized primary directions."""
    recovered_direction = recovered.rotate_vector(vector=primary)
    synthesized_direction = synthesized.rotate_vector(vector=primary)
    dot = float(np.clip(np.dot(recovered_direction, synthesized_direction), -1.0, 1.0))
    return float(np.arccos(dot))


def _primary_direction(*, skeleton: SkeletonDefinition, name: str) -> FloatArray:
    """The segment's primary direction in its own frame, signed by the declared axis.

    This is the direction hydration actually pins, and it is not the bare unit axis: a
    segment whose primary landmark is off the coordinate axis - the clavicle, whose
    acromion is posterior rather than straight lateral - has a primary direction with more
    than one nonzero component, and checking the bare axis would measure its free roll
    instead of its direction.
    """
    segment = skeleton.segments[name]
    primary_position = skeleton.landmarks[
        segment.frame_definition.primary_point_name
    ].local_position.array
    norm = float(np.linalg.norm(primary_position))
    return float(segment.frame_definition.primary_axis.sign) * primary_position / norm


def _direction_errors(
    *,
    skeleton: SkeletonDefinition,
    world_orientations: dict[str, RotationQuaternion],
    hydrated: SkeletonPose,
) -> dict[str, float]:
    """Every segment's primary-direction error, keyed by segment name."""
    return {
        name: _direction_error(
            recovered=hydrated.segment_poses[name].orientation,
            synthesized=world_orientations[name],
            primary=_primary_direction(skeleton=skeleton, name=name),
        )
        for name in world_orientations
    }


def test_round_trip_recovers_the_pose_exactly_without_noise() -> None:
    skeleton, world_orientations, hydrated = _synthesize_and_hydrate(
        noise_scale=0.0, seed=7
    )
    errors = _direction_errors(
        skeleton=skeleton, world_orientations=world_orientations, hydrated=hydrated
    )
    for name, error in errors.items():
        assert error < 1e-6, f"{name}: direction error {error:.2e} rad"

    # Rigid-fit segments recover the full orientation, not just the direction.
    for name, pose in hydrated.segment_poses.items():
        if pose.solved_by is not PoseSolution.RIGID_FIT:
            continue
        dot = world_orientations[name].dot(other=pose.orientation)
        assert abs(dot) > 1.0 - 1e-6, f"{name}: orientation dot {dot:.6f}"


def test_round_trip_is_robust_to_landmark_noise() -> None:
    skeleton, world_orientations, hydrated = _synthesize_and_hydrate(
        noise_scale=2.0, seed=11
    )
    direction_errors = _direction_errors(
        skeleton=skeleton, world_orientations=world_orientations, hydrated=hydrated
    )

    # Two regimes, split by length. A long bone's direction is pinned by a long lever arm,
    # so a couple of millimetres of noise costs only a degree or two. A short finger bone
    # (16-45 mm) has almost no lever arm, so the same noise swings its direction by tens of
    # degrees; those are checked only for boundedness (no 180 degree flips).
    long_errors = [
        error
        for name, error in direction_errors.items()
        if skeleton.segments[name].length >= SHORT_SEGMENT_LENGTH_MILLIMETRES
    ]
    short_errors = [
        error
        for name, error in direction_errors.items()
        if skeleton.segments[name].length < SHORT_SEGMENT_LENGTH_MILLIMETRES
    ]
    assert np.mean(long_errors) < 0.05, f"long mean {np.mean(long_errors):.4f} rad"
    assert max(long_errors) < 0.2, f"long worst {max(long_errors):.3f} rad"
    assert max(short_errors) < 0.8, f"short worst {max(short_errors):.3f} rad"

    # The rigid-fit segments keep their full orientation under noise.
    for name, pose in hydrated.segment_poses.items():
        if pose.solved_by is not PoseSolution.RIGID_FIT:
            continue
        dot = world_orientations[name].dot(other=pose.orientation)
        assert abs(dot) > 0.99, f"{name}: orientation dot {dot:.4f}"


def test_every_segment_reports_how_it_was_solved() -> None:
    skeleton, _world_orientations, hydrated = _synthesize_and_hydrate(
        noise_scale=0.0, seed=3
    )
    assert len(hydrated.segment_poses) == len(skeleton.segments)
    for name, pose in hydrated.segment_poses.items():
        expected = (
            PoseSolution.RIGID_FIT
            if skeleton.segments[name].supports_rigid_fit
            else PoseSolution.DIRECTION
        )
        assert pose.solved_by is expected, f"{name} solved by {pose.solved_by}"
    assert hydrated.segment_names_with_free_roll, (
        "the shipped skeleton has direction-only segments, so this must not be empty"
    )
