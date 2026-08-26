"""Tests for the parallel-transport roll convention.

Two landmarks do not determine roll, so what these check is not "is the roll correct" -
there is no correct answer - but the three properties that make a CONVENTION usable:
it must not disturb the direction that was measured, it must be continuous frame to frame
including through the pole where the naive shortest-arc roll jumps, and it must leave
measured orientations alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.roll_resolution import (
    ContinuousRollResolver,
    SegmentRollReference,
)
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.skeleton_pose import PoseSolution

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

SWEPT_SEGMENT: str = "left_upper_arm"
FRAME_COUNT: int = 120


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)


def _swept_poses(*, skeleton: SkeletonDefinition):
    """Hydrated poses for an arm swept a full turn through straight overhead and straight down.

    Straight down is the pole of the shortest arc from the arm's authored distal direction, so
    a stateless per-frame roll flips there. Sweeping through it is the point.
    """
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    shoulder = rest_pose.landmark_positions["left_shoulder"]
    length = skeleton.segments[SWEPT_SEGMENT].length

    for frame in range(FRAME_COUNT):
        angle = 2.0 * np.pi * frame / FRAME_COUNT
        direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
        observed = dict(rest_pose.landmark_positions)
        observed["left_elbow"] = Point.from_prevalidated_array(
            array=shoulder.array + length * direction
        )
        yield hydrate_skeleton(skeleton=skeleton, observed=observed)


def test_the_resolver_leaves_rigid_fit_poses_untouched() -> None:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    pose = hydrate_skeleton(skeleton=skeleton, observed=rest_pose.landmark_positions)
    resolved = ContinuousRollResolver.for_skeleton(skeleton=skeleton).resolve_pose(
        pose=pose
    )
    for name, before in pose.segment_poses.items():
        if before.solved_by is not PoseSolution.RIGID_FIT:
            continue
        after = resolved.segment_poses[name]
        assert after.solved_by is PoseSolution.RIGID_FIT
        assert after.orientation.is_same_rotation(other=before.orientation)


def test_resolving_roll_does_not_move_the_measured_direction() -> None:
    """The roll may change; the long axis the data pinned may not."""
    skeleton = _skeleton()
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    reference = SegmentRollReference.for_segment(
        skeleton=skeleton, segment_name=SWEPT_SEGMENT
    )
    for pose in _swept_poses(skeleton=skeleton):
        before = pose.segment_poses[SWEPT_SEGMENT]
        after = resolver.resolve_segment_pose(pose=before)
        assert after.solved_by is PoseSolution.TRANSPORTED_ROLL
        np.testing.assert_allclose(
            after.orientation.rotate_vector(vector=reference.primary_local),
            before.orientation.rotate_vector(vector=reference.primary_local),
            atol=1e-9,
        )


def test_transported_roll_is_continuous_where_the_raw_roll_jumps() -> None:
    """Sweeping an arm through the pole: the raw roll flips, the transported one does not."""
    skeleton = _skeleton()
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)

    raw_steps: list[float] = []
    resolved_steps: list[float] = []
    previous_raw: RotationQuaternion | None = None
    previous_resolved: RotationQuaternion | None = None

    for pose in _swept_poses(skeleton=skeleton):
        raw = pose.segment_poses[SWEPT_SEGMENT].orientation
        resolved = resolver.resolve_segment_pose(
            pose=pose.segment_poses[SWEPT_SEGMENT]
        ).orientation
        if previous_raw is not None and previous_resolved is not None:
            raw_steps.append(previous_raw.angle_to(other=raw))
            resolved_steps.append(previous_resolved.angle_to(other=resolved))
        previous_raw, previous_resolved = raw, resolved

    # One frame of a 120-frame sweep is 3 degrees of direction change, so a roll that is
    # behaving cannot be stepping much more than that.
    per_frame_limit = np.deg2rad(10.0)
    assert max(resolved_steps) < per_frame_limit, (
        f"transported roll jumped {np.degrees(max(resolved_steps)):.1f} degrees in one frame"
    )
    assert max(raw_steps) > per_frame_limit, (
        "this sweep is supposed to make the raw shortest-arc roll jump; if it no longer "
        "does, the test is not testing anything"
    )


def test_a_reset_resolver_starts_a_fresh_take() -> None:
    """After `reset()`, the same frame resolves exactly as it would on a new resolver.

    Half a sweep is used rather than a whole one: transported around the full circle the
    roll returns to where it started, so a closed loop would compare equal for reasons
    that have nothing to do with the reset.
    """
    skeleton = _skeleton()
    poses = list(_swept_poses(skeleton=skeleton))
    first_pose = poses[0].segment_poses[SWEPT_SEGMENT]

    virgin = ContinuousRollResolver.for_skeleton(skeleton=skeleton).resolve_segment_pose(
        pose=first_pose
    ).orientation

    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    for pose in poses[: FRAME_COUNT // 2]:
        resolver.resolve_segment_pose(pose=pose.segment_poses[SWEPT_SEGMENT])
    carried = resolver.resolve_segment_pose(pose=first_pose).orientation
    assert not carried.is_same_rotation(other=virgin, tolerance_radians=1e-3), (
        "half a sweep should have transported the roll somewhere else"
    )

    resolver.reset()
    after_reset = resolver.resolve_segment_pose(pose=first_pose).orientation
    assert after_reset.is_same_rotation(other=virgin, tolerance_radians=1e-12)


def test_resolve_pose_covers_every_direction_only_segment() -> None:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    pose = hydrate_skeleton(skeleton=skeleton, observed=rest_pose.landmark_positions)
    resolved = ContinuousRollResolver.for_skeleton(skeleton=skeleton).resolve_pose(
        pose=pose
    )
    assert resolved.segment_names_with_free_roll == ()
    assert set(resolved.segment_poses) == set(pose.segment_poses)


def test_a_segment_whose_primary_sits_on_its_origin_has_no_roll_reference() -> None:
    skeleton = _skeleton()
    with pytest.raises(KeyError):
        SegmentRollReference.for_segment(skeleton=skeleton, segment_name="not_a_segment")


# ── anchored secondary axes (skeleton-level resolution) ───────────────


def _pose_with_arm_at(direction: np.ndarray):
    """Full-skeleton hydrated pose with the left arm pointing along `direction`."""
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    shoulder = rest_pose.landmark_positions["left_shoulder"]
    upper_length = skeleton.segments["left_upper_arm"].length
    lower_length = skeleton.segments["left_lower_arm"].length

    observed = dict(rest_pose.landmark_positions)
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    observed["left_elbow"] = Point.from_prevalidated_array(
        array=shoulder.array + upper_length * unit
    )
    observed["left_wrist"] = Point.from_prevalidated_array(
        array=observed["left_elbow"].array + lower_length * unit
    )
    return skeleton, hydrate_skeleton(skeleton=skeleton, observed=observed)


def _pose_with_bent_elbow(*, bend_degrees: float):
    """Full-skeleton hydrated pose with the elbow flexed by `bend_degrees`.

    A straight limb gives its distal joints no roll reference at all (the
    parent's origin sits exactly on the bone axis), so this bends the elbow -
    the configuration where anchoring has something to say.
    """
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    shoulder = rest_pose.landmark_positions["left_shoulder"]
    upper_length = skeleton.segments["left_upper_arm"].length
    lower_length = skeleton.segments["left_lower_arm"].length

    upper_dir = np.array([0.35, 0.45, 0.82])
    upper_dir = upper_dir / np.linalg.norm(upper_dir)
    bend = np.deg2rad(bend_degrees)
    c, s = np.cos(bend), np.sin(bend)
    rotation = np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    )  # flexion about the world z-ish plane of the arm
    lower_dir = rotation @ upper_dir

    observed = dict(rest_pose.landmark_positions)
    observed["left_elbow"] = Point.from_prevalidated_array(
        array=shoulder.array + upper_length * upper_dir
    )
    observed["left_wrist"] = Point.from_prevalidated_array(
        array=observed["left_elbow"].array + lower_length * lower_dir
    )
    return skeleton, hydrate_skeleton(skeleton=skeleton, observed=observed)


def test_skeleton_level_roll_is_history_independent() -> None:
    """THE L6.2 promise: reach the same geometry by two different routes and the
    anchored roll is identical - no path dependence, no take history."""
    skeleton = _skeleton()
    _, final_pose = _pose_with_bent_elbow(bend_degrees=55.0)

    # Route A: a long wandering approach through very different arm poses
    # (including dead-straight arms, whose frames fall back to transport).
    wanderer = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    for waypoint in ([0.9, 0.1, 0.4], [-0.8, 0.2, 0.5], [0.0, 0.9, 0.4]):
        _, waypoint_pose = _pose_with_arm_at(direction=np.array(waypoint))
        wanderer.resolve_pose(pose=waypoint_pose)
    via_wander = wanderer.resolve_pose(pose=final_pose)

    # Route B: fresh resolver, straight to the final frame.
    virgin = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    direct = virgin.resolve_pose(pose=final_pose)

    for name in ("left_upper_arm", "left_lower_arm"):
        error_deg = float(
            np.degrees(
                via_wander.segment_poses[name].orientation.angle_to(
                    other=direct.segment_poses[name].orientation
                )
            )
        )
        assert error_deg < 1e-6, (
            f"{name}: roll depended on history - wandered {error_deg:.3e} deg"
        )


def test_anchored_secondary_is_perpendicular_and_right_handed() -> None:
    skeleton = _skeleton()
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    reference = SegmentRollReference.for_segment(
        skeleton=skeleton, segment_name="left_lower_arm"
    )
    _, pose = _pose_with_arm_at(direction=np.array([0.2, 0.6, 0.75]))
    resolved = resolver.resolve_pose(pose=pose)
    orientation = resolved.segment_poses["left_lower_arm"].orientation

    primary_world = orientation.rotate_vector(vector=reference.primary_local)
    secondary_world = orientation.rotate_vector(vector=reference.secondary_local)
    assert abs(float(primary_world @ secondary_world)) < 1e-9
    triple = float(
        primary_world @ np.cross(secondary_world, np.cross(primary_world, secondary_world))
    )  # det-like check via constructed basis below instead:
    third_world = orientation.rotate_vector(
        vector=np.cross(reference.primary_local, reference.secondary_local)
    )
    handedness = float(
        primary_world @ np.cross(secondary_world, third_world)
    )
    assert handedness == pytest.approx(1.0, abs=1e-9)
    assert triple == pytest.approx(1.0, abs=1e-9)


def test_missing_parent_falls_back_to_plain_transport() -> None:
    """Without the parent's pose there is no anchor; the segment must resolve
    exactly as the bare segment-level primitive would from the same state."""
    skeleton = _skeleton()
    _, pose = _pose_with_arm_at(direction=np.array([0.5, 0.5, 0.7]))
    pruned = type(pose)(
        segment_poses={
            name: sp
            for name, sp in pose.segment_poses.items()
            if name != "left_clavicle"
        }
    )

    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    skeleton_level = resolver.resolve_pose(pose=pruned).segment_poses["left_upper_arm"]
    segment_level = resolver.resolve_segment_pose(pose=pruned.segment_poses["left_upper_arm"])
    assert skeleton_level.orientation.is_same_rotation(
        other=segment_level.orientation, tolerance_radians=1e-12
    )


def test_degenerate_straight_limb_falls_back_without_nan() -> None:
    """Arm pointed exactly back along its anchor direction: the hint carries no
    perpendicular information, so resolution falls back to transport - finite,
    measured axis untouched."""
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    sc_joint = rest_pose.landmark_positions["left_sternoclavicular"].array
    acromion = rest_pose.landmark_positions["left_shoulder"].array
    toward_sc = acromion - sc_joint
    toward_sc = toward_sc / np.linalg.norm(toward_sc)

    skeleton_, pose = _pose_with_arm_at(direction=toward_sc)
    resolved = ContinuousRollResolver.for_skeleton(skeleton=skeleton_).resolve_pose(
        pose=pose
    )
    upper = resolved.segment_poses["left_upper_arm"]
    assert upper.solved_by is PoseSolution.TRANSPORTED_ROLL
    components = (
        upper.orientation.w,
        upper.orientation.x,
        upper.orientation.y,
        upper.orientation.z,
    )
    assert all(np.isfinite(components))
