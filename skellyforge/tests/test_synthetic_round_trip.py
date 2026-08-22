"""Round-trip synthetic ground-truth tests for skeleton hydration.

Generate a pose with the forward model (build_rest_pose), project it to landmark positions,
corrupt the landmarks, and solve back with hydrate_skeleton. The recovered orientations
must match the ones that were synthesized - exactly when there is no noise, and within a
noise-scaled tolerance when there is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.math.kinematics.coordinate_frame_ops import primary_axis_unit
from skellyforge.core.skeleton_parts.rest_pose import build_rest_pose
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton_parts.skeleton_hydration import hydrate_skeleton

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


def _read_rest_pose_tree() -> tuple[dict, dict, dict]:
    """The rest pose's parent / connect_at / relative-orientation maps, from its YAML."""
    document = yaml.safe_load(REST_POSE_YAML_PATH.read_text(encoding="utf-8"))
    parents: dict[str, str | None] = {}
    connect_ats: dict[str, str | None] = {}
    orientations: dict[str, RotationQuaternion] = {}
    for name, entry in document["segments"].items():
        parents[name] = entry.get("parent")
        connect_ats[name] = entry.get("connect_at")
        orientation = entry.get("orientation", [1.0, 0.0, 0.0, 0.0])
        orientations[name] = RotationQuaternion.from_components(
            w=float(orientation[0]),
            x=float(orientation[1]),
            y=float(orientation[2]),
            z=float(orientation[3]),
        )
    return parents, connect_ats, orientations


def _synthesize_and_hydrate(noise_scale: float, seed: int):
    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    parents, connect_ats, rest_orientations = _read_rest_pose_tree()

    rng = np.random.default_rng(seed)
    # Perturb every segment's relative orientation: a synthetic "joint movement".
    synthesized = {
        name: RotationQuaternion.from_rotation_vector(rng.normal(scale=0.3, size=3))
        * orientation
        for name, orientation in rest_orientations.items()
    }

    # Forward kinematics: the ground-truth pose -> landmark positions.
    world_orientations, _world_origins, true_landmarks = build_rest_pose(
        skeleton=skeleton,
        parents=parents,
        connect_ats=connect_ats,
        orientations=synthesized,
    )

    observed = {
        name: Point.from_prevalidated_array(
            array=point.array + rng.normal(scale=noise_scale, size=3)
        )
        for name, point in true_landmarks.items()
    }

    hydrated = hydrate_skeleton(skeleton=skeleton, observed=observed)
    return skeleton, world_orientations, hydrated


def _direction_error(recovered: RotationQuaternion, synthesized: RotationQuaternion, primary) -> float:
    recovered_direction = recovered.rotate_vector(primary)
    synthesized_direction = synthesized.rotate_vector(primary)
    dot = float(np.clip(np.dot(recovered_direction, synthesized_direction), -1.0, 1.0))
    return float(np.arccos(dot))


def test_round_trip_recovers_the_pose_exactly_without_noise() -> None:
    skeleton, world_orientations, hydrated = _synthesize_and_hydrate(noise_scale=0.0, seed=7)

    for name in world_orientations:
        primary = primary_axis_unit(skeleton.segments[name].frame_definition.primary_axis)
        error = _direction_error(hydrated.segment_poses[name].orientation, world_orientations[name], primary)
        assert error < 1e-6, f"{name}: direction error {error:.2e} rad"

    # Rigid-fit segments recover the full orientation, not just the direction.
    for name in ("skull", "pelvis", "chest"):
        dot = world_orientations[name].dot(hydrated.segment_poses[name].orientation)
        assert abs(dot) > 1.0 - 1e-6, f"{name}: orientation dot {dot:.6f}"


def test_round_trip_is_robust_to_landmark_noise() -> None:
    skeleton, world_orientations, hydrated = _synthesize_and_hydrate(noise_scale=2.0, seed=11)

    direction_errors: dict[str, float] = {}
    for name in world_orientations:
        primary = primary_axis_unit(skeleton.segments[name].frame_definition.primary_axis)
        direction_errors[name] = _direction_error(
            hydrated.segment_poses[name].orientation, world_orientations[name], primary
        )

    # Two regimes, split by length. A long bone's direction is pinned by a long lever arm, so a
    # couple of millimetres of noise costs only a degree or two. A short finger bone (16-45 mm)
    # has almost no lever arm, so the same noise swings its direction by tens of degrees; those
    # are checked only for boundedness (no 180 degree flips).
    long_errors = [e for name, e in direction_errors.items() if skeleton.segments[name].length >= 50.0]
    short_errors = [e for name, e in direction_errors.items() if skeleton.segments[name].length < 50.0]
    assert np.mean(long_errors) < 0.05, f"long mean direction error {np.mean(long_errors):.4f} rad"
    assert max(long_errors) < 0.2, f"long worst direction error {max(long_errors):.3f} rad"
    assert max(short_errors) < 0.8, f"short worst direction error {max(short_errors):.3f} rad"

    # The rigid-fit segments keep their full orientation under noise.
    for name in ("skull", "pelvis", "chest"):
        dot = world_orientations[name].dot(hydrated.segment_poses[name].orientation)
        assert abs(dot) > 0.99, f"{name}: orientation dot {dot:.4f}"
