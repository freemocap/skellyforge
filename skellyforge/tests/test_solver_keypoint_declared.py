"""The keypoint-declared solver: every driven segment solvable, twist tiers."""

import numpy as np
import pytest

from skellyforge.kinematics.orientation_solver import (
    FrameOrientationResult,
    solve_frame_orientations,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.reference_geometry import (
    build_reference_geometry,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)

_HEIGHT_MM = 1700.0


@pytest.fixture(scope="module")
def rig():
    """The composed human + its T-pose reference + the rest keypoint map."""
    human = compose_standard_human()
    lengths = {s.name: s.length_ratio * _HEIGHT_MM for s in human.segments}
    reference = build_reference_geometry(list(human.segments), lengths)
    return human, reference


def _bent_keypoints(reference, *, straight_arms=True):
    """A plausible standing pose: the rest pose with a few joints moved.

    Differential, deliberately: the elbows bend (forearm rotated), the knees
    flex slightly, the head yaws, and the right arm differs from the left —
    so no uniform bend and no coaxial pair can hide an operand-order bug
    (doc 14 §2). With ``straight_arms`` the elbows stay collinear to exercise
    the singularity gate.
    """
    keypoints = {name: pos.copy() for name, pos in reference.keypoints.items()}
    # nose is an off-chain keypoint with no schematic rest position — a live
    # pose supplies it (anterior of head_center), so the head/neck/face
    # long-axis solves have an endpoint; it is yawed below like head_vertex.
    if "head_center" in keypoints:
        keypoints["nose"] = keypoints["head_center"] + np.array([60.0, 0.0, 0.0])
    # head yaw: nose + head_vertex rotate ~20° about Z around head_center
    if "head_center" in keypoints and "nose" in keypoints:
        origin = keypoints["head_center"]
        theta = np.deg2rad(20.0)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        for name in ("nose", "head_vertex"):
            if name in keypoints:
                keypoints[name] = origin + rot @ (keypoints[name] - origin)
    if not straight_arms:
        for side in ("left", "right"):
            elbow = f"{side}_elbow"
            wrist = f"{side}_wrist"
            if elbow in keypoints and wrist in keypoints:
                # rotate the forearm +30° about the local anterior axis:
                # approximate with a rotation of the wrist about Z through the elbow
                origin = keypoints[elbow]
                theta = np.deg2rad(30.0)
                rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                keypoints[wrist] = origin + rot @ (keypoints[wrist] - origin)
    return keypoints


def test_every_segment_produces_an_orientation(rig):
    human, reference = rig
    keypoints = _bent_keypoints(reference)
    # The nose rest position is off-chain (no segment placed it) — the fixture
    # supplies it so the face bones (left_eye/right_eye/jaw) have a long-axis
    # endpoint to solve against this frame.
    keypoints["nose"] = keypoints["head_center"] + np.array([60.0, 0.0, 0.0])
    result = solve_frame_orientations(
        human, reference.segments, keypoints, timestamp_seconds=1.0
    )
    assert set(result.world_quaternions) == set(human.segment_names)  # 55 of 55


def test_leaf_segments_are_solvable(rig):
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, _bent_keypoints(reference), timestamp_seconds=1.0
    )
    for name in ("head", "left_hand", "left_toes"):
        assert name in result.world_quaternions


def test_multi_child_segment_uses_its_declared_long_axis_keypoint(rig):
    human, reference = rig
    keypoints = _bent_keypoints(reference)
    # move trunk_center forward: hips' declared long axis is trunk_center, not
    # its first child (spine)
    keypoints["trunk_center"] = keypoints["trunk_center"] + np.array([40.0, 0.0, 0.0])
    result = solve_frame_orientations(
        human, reference.segments, keypoints, timestamp_seconds=1.0
    )
    q = RotationQuaternion(*result.world_quaternions["hips"].tolist())
    live_dir = keypoints["trunk_center"] - keypoints["hips_center"]
    live_dir = live_dir / np.linalg.norm(live_dir)
    assert np.allclose(q.rotate_vector(reference.segments["hips"].basis[0]),
                       live_dir, atol=1e-9)


def test_coincident_live_keypoints_skip_the_segment_not_raise(rig):
    human, reference = rig
    keypoints = _bent_keypoints(reference)
    keypoints["head_center"] = keypoints["neck_center"].copy()  # neck degenerates
    result = solve_frame_orientations(
        human, reference.segments, keypoints, timestamp_seconds=1.0
    )
    assert "neck" not in result.world_quaternions
    assert "hips" in result.world_quaternions


def test_degenerate_declaration_raises_at_load_not_at_solve():
    from skellyforge.skellymodels.standard_human.segment_definition import (
        ParentAttachment,
        SegmentDefinition,
    )

    with pytest.raises(ValueError, match="origin_keypoint and long_axis_keypoint"):
        SegmentDefinition(
            name="bad", parent=None, parent_attachment=ParentAttachment.ORIGIN,
            origin_keypoint="same", long_axis_keypoint="same", twist_keypoint=None,
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        )


def test_declared_twist_keypoint_resolves_roll_undamped(rig):
    human, reference = rig
    keypoints = _bent_keypoints(reference, straight_arms=False)
    result = solve_frame_orientations(
        human, reference.segments, keypoints, timestamp_seconds=1.0
    )
    # the bent elbow moves the wrist off the humerus axis: upper_arm's roll is
    # resolved from the declared twist keypoint — measured, not damped
    assert "left_upper_arm" not in result.damping_states
    q = RotationQuaternion(*result.world_quaternions["left_upper_arm"].tolist())
    ref = reference.segments["left_upper_arm"]
    origin = keypoints["left_shoulder"]
    live_long = keypoints["left_elbow"] - origin
    live_long = live_long / np.linalg.norm(live_long)
    live_twist = keypoints["left_wrist"] - origin
    live_twist = live_twist / np.linalg.norm(live_twist)
    # the solved rotation carries the reference basis onto the live basis
    assert np.allclose(q.rotate_vector(ref.basis[0]), live_long, atol=1e-9)


def test_singularity_gate_degrades_straight_limbs_to_damped(rig):
    human, reference = rig
    keypoints = _bent_keypoints(reference, straight_arms=True)
    result = solve_frame_orientations(
        human, reference.segments, keypoints, timestamp_seconds=1.0
    )
    assert "left_upper_arm" in result.damping_states
    assert "right_upper_arm" in result.damping_states


def test_twistless_segment_holds_and_damps(rig):
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, _bent_keypoints(reference), timestamp_seconds=1.0
    )
    assert "left_index_proximal" in result.damping_states


def test_composition_round_trip_recomposes_parent_and_local(rig):
    # doc 14 §2: recompose(parent_world, local) == child_world for every
    # parent/child pair — this is what catches operand-order bugs.
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, _bent_keypoints(reference), timestamp_seconds=1.0
    )
    for segment in human.segments:
        if segment.parent is None or segment.parent not in result.world_quaternions:
            continue
        if segment.name not in result.world_quaternions:
            continue
        parent_world = RotationQuaternion(*result.world_quaternions[segment.parent].tolist())
        local = RotationQuaternion(*result.local_quaternions[segment.name].tolist())
        child_world = RotationQuaternion(*result.world_quaternions[segment.name].tolist())
        recomposed = parent_world * local
        assert np.allclose(
            [recomposed.w, recomposed.x, recomposed.y, recomposed.z],
            [child_world.w, child_world.x, child_world.y, child_world.z],
            atol=1e-9,
        ), segment.name


def test_damping_continuity_across_frames(rig):
    # The gated segment's damped output must LAG the raw target, not jump to it:
    # solve frame 2 both ways — with the previous frame (damped) and fresh
    # (raw, seeded at rest) — and assert the damped step is smaller.
    #
    # The gated upper arm's raw target is swing-only (its twist is collinear at
    # a straight arm and never resolves), so the wrist move in the original
    # review snippet was a no-op (raw_step == 0). To make the swing ACTUALLY
    # change while the arm stays gated, pivot the whole raised arm sideways:
    # move the elbow and wrist together perpendicular to the arm axis (+X). The
    # arm keeps its straight (collinear) forearm, so it stays in the damped
    # tier, but its long-axis direction changes — a real target jump that the
    # critically-damped filter must lag.
    human, reference = rig
    keypoints2 = _bent_keypoints(reference, straight_arms=True)
    for keypoint in ("left_elbow", "left_wrist"):
        keypoints2[keypoint] = keypoints2[keypoint] + np.array([15.0, 0.0, 0.0])
    first = solve_frame_orientations(
        human, reference.segments, _bent_keypoints(reference, straight_arms=True),
        timestamp_seconds=1.0,
    )
    damped = solve_frame_orientations(
        human, reference.segments, keypoints2,
        timestamp_seconds=1.1, previous_result=first,
    )
    raw = solve_frame_orientations(
        human, reference.segments, keypoints2,
        timestamp_seconds=1.1, previous_result=None,
    )
    assert "left_upper_arm" in damped.damping_states

    def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
        dot = abs(float(np.dot(a, b)))
        return float(np.arccos(min(1.0, dot)))

    q_first = first.world_quaternions["left_upper_arm"]
    q_damped = damped.world_quaternions["left_upper_arm"]
    q_raw = raw.world_quaternions["left_upper_arm"]

    damped_step = _angle_between(q_first, q_damped)
    raw_step = _angle_between(q_first, q_raw)
    # the raw target moved (the fixture is a real jump), and damping lags it
    assert raw_step > 1e-6
    assert damped_step < raw_step


def test_identity_at_t_pose(rig):
    # doc 14 §4: feeding the model's own rest geometry as live input, every
    # solved segment must return identity — this is the identity == T-pose
    # contract every downstream consumer assumes.
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, reference.keypoints, timestamp_seconds=1.0
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    for name, q in result.world_quaternions.items():
        dev = min(np.linalg.norm(q - identity), np.linalg.norm(q + identity))
        assert dev < 1e-9, f"{name} deviates from identity at the T-pose by {dev}"
    for name, q in result.local_quaternions.items():
        dev = min(np.linalg.norm(q - identity), np.linalg.norm(q + identity))
        assert dev < 1e-9, f"{name} local deviates from identity at the T-pose by {dev}"


def test_singularity_gate_threshold_boundary(rig):
    # The gate switches at ~5° between the long axis and the twist direction.
    # The long axis of a raised arm is +Y, so rotating the wrist about Z through
    # the elbow deflects the wrist-vs-long-axis direction at ~0.44x the bend —
    # the gate crossing is a bend of ~11.4°, NOT 5°. A 10° bend keeps the wrist
    # direction inside the gate → damped; a 13° bend moves it outside → the
    # declared twist keypoint resolves the roll (matches the comments).
    from skellyforge.kinematics.orientation_solver import (
        _SINGULARITY_THRESHOLD_RAD,
    )

    human, reference = rig

    def _solve_with_bend(bend_deg: float):
        keypoints = {n: p.copy() for n, p in reference.keypoints.items()}
        origin = keypoints["left_elbow"]
        theta = np.deg2rad(bend_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
        keypoints["left_wrist"] = origin + rot @ (keypoints["left_wrist"] - origin)
        return solve_frame_orientations(
            human, reference.segments, keypoints, timestamp_seconds=1.0
        )

    inside = _solve_with_bend(10.0)   # within ~5° → gated
    outside = _solve_with_bend(13.0)  # outside ~5° → declared twist resolves
    assert np.isclose(_SINGULARITY_THRESHOLD_RAD, np.deg2rad(5.0))
    assert "left_upper_arm" in inside.damping_states
    assert "left_upper_arm" not in outside.damping_states
