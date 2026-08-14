"""The landmark-declared solver: every driven segment solvable, twist tiers."""

import numpy as np
import pytest

from skellyforge.kinematics.coordinate_frame_ops import _AXIS_TO_INDEX
from skellyforge.kinematics.orientation_solver import (
    FrameOrientationResult,
    solve_frame_orientations,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.reference_geometry import (
    build_reference_geometry,
)
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    AxisKind,
    ParentAttachment,
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)

_HEIGHT_MM = 1700.0


@pytest.fixture(scope="module")
def rig():
    """The composed human + its T-pose reference + the rest landmark map."""
    human = compose_standard_human()
    lengths = {s.name: s.length_ratio * _HEIGHT_MM for s in human.segments}
    reference = build_reference_geometry(list(human.segments), lengths)
    return human, reference


def _bent_landmarks(reference, *, straight_arms=True):
    """A plausible standing pose: the rest pose with a few joints moved.

    Differential, deliberately: the elbows bend (forearm rotated), the knees
    flex slightly, the head yaws, and the right arm differs from the left —
    so no uniform bend and no coaxial pair can hide an operand-order bug
    (doc 14 §2). With ``straight_arms`` the elbows stay collinear to exercise
    the singularity gate.
    """
    landmarks = {name: pos.copy() for name, pos in reference.landmarks.items()}
    # nose is an off-chain landmark with no schematic rest position — a live
    # pose supplies it (anterior of head_center), so the head/neck/face
    # long-axis solves have an endpoint; it is yawed below like head_vertex.
    if "head_center" in landmarks:
        landmarks["nose"] = landmarks["head_center"] + np.array([60.0, 0.0, 0.0])
    # the face-detail segments' long-axis landmarks (ears, mouth corners) have
    # no schematic rest position either — a live pose supplies them, placed so
    # each segment's origin→long-axis direction is non-degenerate this frame.
    if "head_center" in landmarks:
        landmarks["left_ear"] = landmarks["head_center"] + np.array([0.0, 40.0, 30.0])
        landmarks["right_ear"] = landmarks["head_center"] + np.array([0.0, -40.0, 30.0])
        landmarks["left_mouth"] = landmarks["head_center"] + np.array([30.0, 20.0, -20.0])
        landmarks["right_mouth"] = landmarks["head_center"] + np.array([30.0, -20.0, -20.0])
    # head yaw: nose + head_vertex rotate ~20° about Z around head_center
    if "head_center" in landmarks and "nose" in landmarks:
        origin = landmarks["head_center"]
        theta = np.deg2rad(20.0)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        for name in ("nose", "head_vertex"):
            if name in landmarks:
                landmarks[name] = origin + rot @ (landmarks[name] - origin)
    if not straight_arms:
        for side in ("left", "right"):
            elbow = f"{side}_elbow"
            wrist = f"{side}_wrist"
            if elbow in landmarks and wrist in landmarks:
                # rotate the forearm +30° about the local anterior axis:
                # approximate with a rotation of the wrist about Z through the elbow
                origin = landmarks[elbow]
                theta = np.deg2rad(30.0)
                rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                landmarks[wrist] = origin + rot @ (landmarks[wrist] - origin)
    return landmarks


def test_every_segment_produces_an_orientation(rig):
    human, reference = rig
    landmarks = _bent_landmarks(reference)
    # The nose rest position is off-chain (no segment placed it) — the fixture
    # supplies it so the face bones (left_eye/right_eye/jaw) and the face-detail
    # segments have a long-axis endpoint to solve against this frame.
    landmarks["nose"] = landmarks["head_center"] + np.array([60.0, 0.0, 0.0])
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    assert set(result.world_quaternions) == set(human.segment_names)  # 60 of 60


def test_leaf_segments_are_solvable(rig):
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, _bent_landmarks(reference), timestamp_seconds=1.0
    )
    for name in ("head", "left_hand", "left_toes"):
        assert name in result.world_quaternions


def test_multi_child_segment_uses_its_declared_exact_axis(rig):
    human, reference = rig
    landmarks = _bent_landmarks(reference)
    # move trunk_center forward: hips' declared exact axis is trunk_center, not
    # its first child (spine); the exact axis is declared on y (basis row 1).
    landmarks["trunk_center"] = landmarks["trunk_center"] + np.array([40.0, 0.0, 0.0])
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    q = RotationQuaternion(*result.world_quaternions["hips"].tolist())
    live_dir = landmarks["trunk_center"] - landmarks["hips_center"]
    live_dir = live_dir / np.linalg.norm(live_dir)
    # the swing maps the reference basis vector NAMED by the exact axis (y →
    # row 1) to the live exact direction.
    assert np.allclose(q.rotate_vector(reference.segments["hips"].basis[1]),
                       live_dir, atol=1e-9)


def test_solver_swing_is_name_driven(rig):
    # A segment whose exact axis is declared on y maps ref basis[1] (ŷ) to the
    # live exact direction; a z-exact segment maps basis[2] (ẑ). This is the
    # name-driven correspondence: never a positional read of basis[0].
    human, reference = rig
    landmarks = _bent_landmarks(reference)
    # hips — exact on y: mapping basis[1]
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    q_hips = RotationQuaternion(*result.world_quaternions["hips"].tolist())
    live_dir = landmarks["trunk_center"] - landmarks["hips_center"]
    live_dir = live_dir / np.linalg.norm(live_dir)
    assert np.allclose(q_hips.rotate_vector(reference.segments["hips"].basis[1]),
                       live_dir, atol=1e-9)
    # the nose face-detail segment — exact on z: mapping basis[2]
    q_nose = RotationQuaternion(*result.world_quaternions["nose"].tolist())
    live_nose = landmarks["nose"] - landmarks["head_center"]
    live_nose = live_nose / np.linalg.norm(live_nose)
    assert np.allclose(q_nose.rotate_vector(reference.segments["nose"].basis[2]),
                       live_nose, atol=1e-9)


def test_z_exact_and_y_exact_segments_both_construct_and_solve(rig):
    # The exact axis may be declared on any of x/y/z with no positional
    # assumption. Build two standalone segments — a y-exact body-like segment
    # and a z-exact face-like segment — and confirm each constructs a frame from
    # its own rest geometry.
    import math

    y_exact = SegmentDefinition(
        name="s_y", parent=None, parent_attachment=ParentAttachment.ORIGIN,
        landmarks=("o", "p", "q"), origin_landmark="o",
        axes=(
            AxisDefinition("y", AxisKind.EXACT, "p"),
            AxisDefinition("z", AxisKind.APPROXIMATE, "q"),
        ),
        rest_rotation=(math.pi / 2, 0.0, 0.0),  # up (+Z)
        length_ratio=0.1,
    )
    z_exact = SegmentDefinition(
        name="s_z", parent=None, parent_attachment=ParentAttachment.ORIGIN,
        landmarks=("o2", "p2", "q2"), origin_landmark="o2",
        axes=(
            AxisDefinition("z", AxisKind.EXACT, "p2"),
            AxisDefinition("x", AxisKind.APPROXIMATE, "q2"),
        ),
        rest_rotation=(0.0, math.pi / 2, 0.0),  # gaze (+X)
        length_ratio=0.1,
    )
    # Build reference geometry for each standalone segment directly.
    for seg in (y_exact, z_exact):
        ref = build_reference_geometry([seg], {seg.name: 100.0})
        basis = ref.segments[seg.name].basis
        assert np.isclose(np.linalg.det(basis), 1.0), seg.name
        # the exact axis's named row is unit
        idx = _AXIS_TO_INDEX[_exact_axis_name(seg)]
        assert np.isclose(np.linalg.norm(basis[idx]), 1.0), seg.name
    # y-exact: ŷ is up (+Z); z-exact: ẑ is gaze (+X)
    y_ref = build_reference_geometry([y_exact], {"s_y": 100.0})
    z_ref = build_reference_geometry([z_exact], {"s_z": 100.0})
    assert np.allclose(y_ref.segments["s_y"].basis[1], (0.0, 0.0, 1.0), atol=1e-6)
    assert np.allclose(z_ref.segments["s_z"].basis[2], (1.0, 0.0, 0.0), atol=1e-6)


def _exact_axis_name(seg) -> str:
    return next(a.axis for a in seg.axes if a.kind is AxisKind.EXACT)


def test_coincident_live_landmarks_skip_the_segment_not_raise(rig):
    human, reference = rig
    landmarks = _bent_landmarks(reference)
    landmarks["head_center"] = landmarks["neck_center"].copy()  # neck degenerates
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    assert "neck" not in result.world_quaternions
    assert "hips" in result.world_quaternions


def test_degenerate_declaration_raises_at_load_not_at_solve():
    from skellyforge.skellymodels.standard_human.segment_definition import (
        AxisDefinition,
        AxisKind,
        ParentAttachment,
        SegmentDefinition,
    )

    with pytest.raises(ValueError, match="zero-length"):
        SegmentDefinition(
            name="bad", parent=None, parent_attachment=ParentAttachment.ORIGIN,
            landmarks=("same", "other"), origin_landmark="same",
            axes=(AxisDefinition("x", AxisKind.EXACT, "same"),),
            rest_rotation=(0.0, 0.0, 0.0), length_ratio=0.1,
        )


def test_declared_approximate_axis_resolves_roll_undamped(rig):
    human, reference = rig
    landmarks = _bent_landmarks(reference)
    # supply a heel off the foot's long axis: the foot's roll is resolved from
    # its declared approximate target (heel) — measured, not damped
    ankle = landmarks["left_ankle"]
    landmarks["left_heel"] = ankle + np.array([-10.0, 0.0, -30.0])
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    assert "left_foot" not in result.damping_states
    q = RotationQuaternion(*result.world_quaternions["left_foot"].tolist())
    ref = reference.segments["left_foot"]
    live_long = landmarks["left_foot_ball"] - ankle
    live_long = live_long / np.linalg.norm(live_long)
    # the solved rotation carries the reference basis onto the live basis —
    # the foot's exact axis is y, so read basis[1] (ŷ) as the exact direction.
    assert np.allclose(q.rotate_vector(ref.basis[1]), live_long, atol=1e-9)


def test_singularity_gate_degrades_straight_limbs_to_damped(rig):
    human, reference = rig
    landmarks = _bent_landmarks(reference, straight_arms=True)
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    assert "left_upper_arm" in result.damping_states
    assert "right_upper_arm" in result.damping_states


def test_twistless_segment_holds_and_damps(rig):
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, _bent_landmarks(reference), timestamp_seconds=1.0
    )
    assert "left_index_proximal" in result.damping_states


def test_composition_round_trip_recomposes_parent_and_local(rig):
    # doc 14 §2: recompose(parent_world, local) == child_world for every
    # parent/child pair — this is what catches operand-order bugs.
    human, reference = rig
    result = solve_frame_orientations(
        human, reference.segments, _bent_landmarks(reference), timestamp_seconds=1.0
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
    landmarks2 = _bent_landmarks(reference, straight_arms=True)
    for landmark in ("left_elbow", "left_wrist"):
        landmarks2[landmark] = landmarks2[landmark] + np.array([15.0, 0.0, 0.0])
    first = solve_frame_orientations(
        human, reference.segments, _bent_landmarks(reference, straight_arms=True),
        timestamp_seconds=1.0,
    )
    damped = solve_frame_orientations(
        human, reference.segments, landmarks2,
        timestamp_seconds=1.1, previous_result=first,
    )
    raw = solve_frame_orientations(
        human, reference.segments, landmarks2,
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
        human, reference.segments, reference.landmarks, timestamp_seconds=1.0
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    for name, q in result.world_quaternions.items():
        dev = min(np.linalg.norm(q - identity), np.linalg.norm(q + identity))
        assert dev < 1e-9, f"{name} deviates from identity at the T-pose by {dev}"
    for name, q in result.local_quaternions.items():
        dev = min(np.linalg.norm(q - identity), np.linalg.norm(q + identity))
        assert dev < 1e-9, f"{name} local deviates from identity at the T-pose by {dev}"


def test_singularity_gate_threshold_boundary(rig):
    # The gate switches at ~5° between the long axis and the approximate-axis
    # direction. The foot's long axis is ankle → foot_ball; rotating the heel
    # off that axis through the ankle deflects the heel-vs-long-axis direction
    # by the same angle. A heel within ~5° of the long axis (collinear) keeps
    # the foot in the damped tier; a heel further out resolves the roll from
    # the declared approximate target.
    from skellyforge.kinematics.orientation_solver import (
        _SINGULARITY_THRESHOLD_RAD,
    )

    human, reference = rig

    def _solve_with_heel_angle(angle_deg: float):
        landmarks = {n: p.copy() for n, p in reference.landmarks.items()}
        ankle = landmarks["left_ankle"]
        theta = np.deg2rad(angle_deg)
        landmarks["left_heel"] = ankle + np.array(
            [np.cos(theta), np.sin(theta), 0.0]
        ) * 50.0
        return solve_frame_orientations(
            human, reference.segments, landmarks, timestamp_seconds=1.0
        )

    inside = _solve_with_heel_angle(2.0)    # within ~5° → gated
    outside = _solve_with_heel_angle(13.0)  # outside ~5° → declared twist resolves
    assert np.isclose(_SINGULARITY_THRESHOLD_RAD, np.deg2rad(5.0))
    assert "left_foot" in inside.damping_states
    assert "left_foot" not in outside.damping_states


def test_head_solves_to_identity_with_anterior_nose(rig):
    # doc 14 §4 for the head specifically. `test_identity_at_t_pose` feeds
    # `reference.landmarks`, which omits the off-chain `nose`, so the head has
    # no usable approximate target and degrades to the damped tier — returning
    # identity TRIVIALLY, blind to a corrupted reference forward axis. At
    # runtime the skull fit supplies a real anterior `nose`, driving the head
    # through the RESOLVED tier against its reference frame. Feed that here: the
    # head must still be identity at the T-pose (world AND local).
    human, reference = rig
    landmarks = {name: pos.copy() for name, pos in reference.landmarks.items()}
    landmarks["nose"] = landmarks["head_center"] + np.array([60.0, 0.0, 0.0])
    result = solve_frame_orientations(
        human, reference.segments, landmarks, timestamp_seconds=1.0
    )
    assert "head" not in result.damping_states  # resolved tier, not damped
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q_world = result.world_quaternions["head"]
    dev_world = min(
        np.linalg.norm(q_world - identity), np.linalg.norm(q_world + identity)
    )
    assert dev_world < 1e-9, f"head world deviates from identity by {dev_world}"
    q_local = result.local_quaternions["head"]
    dev_local = min(
        np.linalg.norm(q_local - identity), np.linalg.norm(q_local + identity)
    )
    assert dev_local < 1e-9, f"head local deviates from identity by {dev_local}"
