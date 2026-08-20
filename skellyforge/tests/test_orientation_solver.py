"""The orientation solver on the standard-human ontology.

Feeds hydrated landmark positions to ``solve_frame_orientations`` and pins the
solver's contract: ``identity == T-pose`` (world AND local), Kabsch round-trip
for the 3+-landmark rigid bodies, the parent/local composition, the two solve
tiers (Kabsch vs. swing+damped), and occlusion-is-data (a missing landmark skips
its segment rather than raising).

The full rigidify -> solve pipeline at the T-pose is pinned separately in
``test_rigidify_identity_at_tpose.py``; this module drives the solver directly
from the T-pose landmark positions.
"""

import numpy as np
import pytest

from skellyforge.kinematics.orientation_solver import (
    SolveState,
    solve_frame_orientations,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.kinematics.tpose import build_standard_human_tpose
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def skeleton_and_tpose():
    skeleton = HumanSkeleton.standard_human()
    tpose = build_standard_human_tpose(skeleton)
    return skeleton, tpose


def _solve(skeleton, tpose, landmarks, *, timestamp_seconds=1.0, state=None):
    return solve_frame_orientations(
        skeleton,
        tpose,
        landmarks,
        timestamp_seconds=timestamp_seconds,
        state=state if state is not None else SolveState(),
    )


def _deviation_from_identity(quaternion: np.ndarray) -> float:
    return float(
        min(
            np.linalg.norm(quaternion - _IDENTITY),
            np.linalg.norm(quaternion + _IDENTITY),
        )
    )


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """A right-handed rotation matrix about a unit axis (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return c * np.eye(3) + s * cross + (1 - c) * np.outer(axis, axis)


def test_identity_at_tpose_world_and_local(skeleton_and_tpose):
    """Feed the model's own rest geometry as live input; every segment solves to
    identity, world AND local -- the contract every downstream consumer relies on."""
    skeleton, tpose = skeleton_and_tpose
    result, _ = _solve(skeleton, tpose, dict(tpose.landmarks))

    assert set(result.world_quaternions) == set(skeleton.segment_names)
    for name, quaternion in result.world_quaternions.items():
        assert _deviation_from_identity(quaternion) < 1e-9, f"{name} world off identity"
    for name, quaternion in result.local_quaternions.items():
        assert _deviation_from_identity(quaternion) < 1e-9, f"{name} local off identity"


def test_every_segment_with_its_landmarks_produces_an_orientation(skeleton_and_tpose):
    """With every landmark present, all 95 segments get a world quaternion."""
    skeleton, tpose = skeleton_and_tpose
    result, _ = _solve(skeleton, tpose, dict(tpose.landmarks))
    assert set(result.world_quaternions) == set(skeleton.segment_names)


def test_kabsch_segment_recovers_a_known_rotation(skeleton_and_tpose):
    """A 3+-landmark rigid body (hips, 7 landmarks) solved by Kabsch returns the
    exact world rotation applied to its rest cloud."""
    skeleton, tpose = skeleton_and_tpose
    pelvis = skeleton.segment("hips")
    pivot = tpose.landmarks[pelvis.origin_landmark.name]
    rotation = _rotation_about_axis(np.array([0.3, 0.7, 0.65]), np.deg2rad(37.0))

    live = {
        landmark.name: pivot + rotation @ (tpose.landmarks[landmark.name] - pivot)
        for landmark in pelvis.landmarks
    }
    result, _ = _solve(skeleton, tpose, live)

    solved = RotationQuaternion(*result.world_quaternions["hips"].tolist())
    expected = RotationQuaternion.from_rotation_matrix(rotation)
    # compare by action on a probe vector (avoids the q ~ -q sign ambiguity)
    for probe in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        assert np.allclose(
            solved.rotate_vector(probe), expected.rotate_vector(probe), atol=1e-9
        )


def test_composition_recomposes_parent_and_local(skeleton_and_tpose):
    """For every solved parent/child pair, parent_world * local == child_world --
    the operand order the renderer and every joint-angle consumer assume."""
    skeleton, tpose = skeleton_and_tpose
    landmarks = _bent_pose(skeleton, tpose)
    result, _ = _solve(skeleton, tpose, landmarks)

    for segment in skeleton.segments:
        if segment.parent is None:
            continue
        if segment.name not in result.world_quaternions:
            continue
        if segment.parent.name not in result.world_quaternions:
            continue
        parent_world = RotationQuaternion(*result.world_quaternions[segment.parent.name].tolist())
        local = RotationQuaternion(*result.local_quaternions[segment.name].tolist())
        child_world = RotationQuaternion(*result.world_quaternions[segment.name].tolist())
        recomposed = parent_world * local
        assert np.allclose(
            [recomposed.w, recomposed.x, recomposed.y, recomposed.z],
            [child_world.w, child_world.x, child_world.y, child_world.z],
            atol=1e-9,
        ), segment.name


def test_kabsch_segment_does_not_damp(skeleton_and_tpose):
    """An over-determined 3+-landmark body solves by Kabsch and never enters the
    damped tier (no twist ambiguity to damp)."""
    skeleton, tpose = skeleton_and_tpose
    _, state = _solve(skeleton, tpose, dict(tpose.landmarks))
    assert "hips" not in state.orientation.damping_states


def test_single_axis_limb_uses_the_damped_tier(skeleton_and_tpose):
    """A 2-landmark segment with only a primary axis (no twist reference) resolves
    its roll through the critically-damped tier -- e.g. the upper arm."""
    skeleton, tpose = skeleton_and_tpose
    _, state = _solve(skeleton, tpose, dict(tpose.landmarks))
    assert "upper_arm.L" in state.orientation.damping_states


def test_missing_primary_target_skips_the_segment_without_raising(skeleton_and_tpose):
    """Occlusion is data: dropping a segment's primary-axis target removes that
    segment from the result (skipped this frame), does not raise, and leaves its
    parent solved."""
    skeleton, tpose = skeleton_and_tpose
    landmarks = dict(tpose.landmarks)
    # left_upper_arm is shoulder -> elbow; drop the elbow (its primary target)
    del landmarks["left_elbow"]
    result, _ = _solve(skeleton, tpose, landmarks)
    assert "upper_arm.L" not in result.world_quaternions
    assert "shoulder.L" in result.world_quaternions  # the parent still solves


def test_orphaned_child_emits_world_but_no_local(skeleton_and_tpose):
    """A child whose parent failed to solve gets a WORLD orientation but NO local:
    emitting the world rotation as a local one (against a parent frame that does
    not exist this frame) would be composed incorrectly by the renderer."""
    skeleton, tpose = skeleton_and_tpose
    landmarks = dict(tpose.landmarks)
    # drop the shoulder: left_upper_arm (shoulder -> elbow) loses its origin and
    # is skipped, while left_lower_arm (elbow -> wrist) still solves.
    del landmarks["left_shoulder"]
    result, _ = _solve(skeleton, tpose, landmarks)
    assert "upper_arm.L" not in result.world_quaternions   # parent skipped
    assert "lower_arm.L" in result.world_quaternions       # child solved (world)
    assert "lower_arm.L" not in result.local_quaternions   # no orphan local


def _bent_pose(skeleton, tpose):
    """A differential standing pose: bend the left forearm and yaw the head, so no
    uniform rotation and no coaxial pair can hide an operand-order bug."""
    landmarks = {name: position.copy() for name, position in tpose.landmarks.items()}

    # left forearm: rotate the wrist ~30 deg about +X through the elbow
    elbow = landmarks["left_elbow"]
    forearm_rotation = _rotation_about_axis(np.array([1.0, 0.0, 0.0]), np.deg2rad(30.0))
    landmarks["left_wrist"] = elbow + forearm_rotation @ (landmarks["left_wrist"] - elbow)

    # head: yaw ~20 deg about +Z through the head's origin
    head = skeleton.segment("head")
    head_origin = landmarks[head.origin_landmark.name]
    head_rotation = _rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.deg2rad(20.0))
    for landmark in head.landmarks:
        landmarks[landmark.name] = (
            head_origin + head_rotation @ (landmarks[landmark.name] - head_origin)
        )
    return landmarks
