"""Identity-at-T-pose through the full ``rigidify_landmarks`` -> solve pipeline.

The realtime loop never sees a complete landmark set: many of the non-root
segment origins are landmarks no tracker mapping can produce (the deep carpals,
the tarsals, the toe joints, the lumbosacral junction). ``rigidify_landmarks``
fabricates those missing origins by walking the parent-edge tree and placing
each child ``length`` away from its corrected parent along a direction.

The ``identity == T-pose`` contract (conventions.md) requires that, at rest, a
fabricated origin lands EXACTLY on its own rest position -- otherwise every
segment downstream of it solves to a non-identity orientation even though the
subject is standing in the reference pose.

This module is the regression gate for the rigidifier's fallback DIRECTION. It
fails when a missing origin is fabricated along a fixed world axis instead of
the segment's own T-pose rest direction: the sparse case then drags the axial
chain, hands, and feet tens to ~180 degrees off identity (see the audit,
2026-08-19, section 2). It passes when the fallback is the per-segment rest
direction.
"""

import numpy as np

from skellyforge.kinematics.orientation_solver import (
    SolveState,
    solve_frame_orientations,
)
from skellyforge.kinematics.skeleton_rigidifier import rigidify_landmarks
from skellyforge.kinematics.tpose import build_standard_human_tpose
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _deviation_from_identity(quaternion: np.ndarray) -> float:
    """Distance to identity, accounting for the double cover (q ~ -q)."""
    return float(
        min(
            np.linalg.norm(quaternion - _IDENTITY),
            np.linalg.norm(quaternion + _IDENTITY),
        )
    )


def _solve_at_rest(skeleton, tpose, landmarks):
    result, _ = solve_frame_orientations(
        skeleton,
        tpose,
        landmarks,
        timestamp_seconds=1.0,
        state=SolveState(),
    )
    return result


def _assert_all_identity(result) -> None:
    world_worst_name, world_worst_dev = max(
        ((name, _deviation_from_identity(q)) for name, q in result.world_quaternions.items()),
        key=lambda item: item[1],
    )
    assert world_worst_dev < 1e-9, (
        f"{world_worst_name} world orientation is {world_worst_dev:.6g} off identity "
        f"at the T-pose"
    )
    local_worst_name, local_worst_dev = max(
        ((name, _deviation_from_identity(q)) for name, q in result.local_quaternions.items()),
        key=lambda item: item[1],
    )
    assert local_worst_dev < 1e-9, (
        f"{local_worst_name} local orientation is {local_worst_dev:.6g} off identity "
        f"at the T-pose"
    )


def test_identity_at_tpose_with_complete_landmarks():
    """Baseline: feed every rest landmark through rigidify + solve -> identity.

    This exercises the pipeline with no fabrication at all (every origin is
    present), so it isolates the solver + rigid-fit from the fallback path. It
    holds regardless of the fallback direction.
    """
    skeleton = HumanSkeleton.standard_human()
    tpose = build_standard_human_tpose(skeleton)

    rigidified = rigidify_landmarks(skeleton, tpose, dict(tpose.landmarks))
    _assert_all_identity(_solve_at_rest(skeleton, tpose, rigidified))


def test_identity_survives_dropping_every_non_root_origin():
    """The gate: drop every non-root segment origin, then require identity.

    Dropping the origins forces ``rigidify_landmarks`` to fabricate each of them
    from its parent -- the same fallback path the realtime loop takes ~49 times a
    frame for the structurally-unobservable carpals/tarsals/toe-joints. At the
    T-pose each fabricated origin must land back on its rest position (0 mm) and
    every segment must solve to identity.

    On the hardcoded ``[0, 1, 0]`` fallback this fails hard: the fabricated
    origins strew along world +Y and the axial chain, hands, and feet come out
    tens-to-180 degrees off identity. With the per-segment rest-direction
    fallback it is exact.
    """
    skeleton = HumanSkeleton.standard_human()
    tpose = build_standard_human_tpose(skeleton)

    dropped_origins = {
        segment.origin_landmark.name
        for segment in skeleton.segments
        if segment.parent is not None
    }
    sparse = {
        name: position.copy()
        for name, position in tpose.landmarks.items()
        if name not in dropped_origins
    }
    # Guard the premise: dropping origins really does remove a large chunk of the
    # landmark set (so this is a meaningful stress of the fallback, not a no-op).
    assert len(sparse) < len(tpose.landmarks)

    rigidified = rigidify_landmarks(skeleton, tpose, sparse)

    displaced_name, displaced_mm = max(
        (
            (name, float(np.linalg.norm(rigidified[name] - rest_position)))
            for name, rest_position in tpose.landmarks.items()
        ),
        key=lambda item: item[1],
    )
    assert displaced_mm < 1e-6, (
        f"rigidify displaced {displaced_name} by {displaced_mm:.1f} mm from its rest "
        f"position at the T-pose"
    )

    _assert_all_identity(_solve_at_rest(skeleton, tpose, rigidified))
