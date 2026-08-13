"""Per-segment orientation solver: declared keypoints → segment orientations.

Reads the composed StandardHuman's segment declarations: each segment names
its origin, long-axis and (optionally) twist keypoints, and the solver
resolves its orientation from THIS FRAME's keypoint positions against the
T-pose reference geometry (identity == T-pose).

Twist is two-tier, and the declaration IS the policy:
1. A declared twist keypoint, usable this frame (not occluded, not within
   ~5° of the long axis — the singularity gate) → the roll resolves from it.
2. Otherwise → damped minimal roll: the reference approximate axis carried
   by the swing rotation, critically damped (the D3/D4 filter).

Segments whose keypoints are missing or numerically coincident this frame are
skipped — occlusion is data, and the load-time validation (segment_definition)
makes a *declared* coincidence impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    build_orthonormal_basis,
    compute_rotation_from_live_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.critically_damped_orientation import (
    CriticallyDampedOrientationState,
    advance_critically_damped_orientation,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion

if TYPE_CHECKING:
    from numpy import float64
    from skellyforge.skellymodels.standard_human.reference_geometry import (
        SegmentReferenceGeometry,
    )
    from skellyforge.skellymodels.standard_human.standard_human_model import (
        StandardHuman,
    )

# Twist directions within this angle (radians) of the long axis cannot build
# a basis — the singularity gate degrades to the damped minimal tier.
_SINGULARITY_THRESHOLD_RAD = np.deg2rad(5.0)
_SINGULARITY_DOT_THRESHOLD = np.cos(_SINGULARITY_THRESHOLD_RAD)  # ≈ 0.996

# The critically damped twist filter's time constant (seconds). Frame-rate
# independent by construction — see critically_damped_orientation.py.
DEFAULT_TWIST_TIME_CONSTANT_SECONDS = 0.05


@dataclass
class FrameOrientationResult:
    """Orientations for every solved segment at one frame.

    Parameters
    ----------
    world_quaternions : dict
        ``{segment_name: (4,) wxyz array}`` — world-frame rotation from the
        T-pose for each SOLVED segment. Unsolved (occluded) segments are
        absent.
    local_quaternions : dict
        ``{segment_name: (4,) wxyz array}`` — parent-relative rotation,
        ``conjugate(world_parent) * world_child`` (D1). The root's local
        equals its world.
    timestamp_seconds : float
        When this frame was solved; the next frame's dt is measured against it.
    damping_states : dict
        Per-segment critically damped filter state for the segments whose roll
        was resolved by the damped minimal tier. Carried into the next frame.
    """

    world_quaternions: dict[str, NDArray[float64]]
    local_quaternions: dict[str, NDArray[float64]]
    timestamp_seconds: float
    damping_states: dict[str, CriticallyDampedOrientationState] = field(
        default_factory=dict
    )


def solve_frame_orientations(
    standard_human: "StandardHuman",
    reference_geometry: dict[str, "SegmentReferenceGeometry"],
    keypoints: dict[str, NDArray[float64]],
    *,
    timestamp_seconds: float,
    previous_result: FrameOrientationResult | None = None,
) -> FrameOrientationResult:
    """Compute per-segment world + local quaternions for one frame.

    Walks segments in authoring order (hierarchy order by construction), so
    parent world quaternions exist when children compose their locals.

    Damping
    -------
    Segments resolved by the damped minimal tier pass through the critically
    damped filter with ``dt = timestamp_seconds - previous.timestamp_seconds``.
    First frame (or a non-advancing clock) seeds at rest with zero velocity.
    """
    timestep_seconds: float | None = None
    if previous_result is not None:
        elapsed = timestamp_seconds - previous_result.timestamp_seconds
        timestep_seconds = elapsed if elapsed > 0.0 else None

    previous_damping_states = (
        previous_result.damping_states if previous_result is not None else {}
    )
    damping_states: dict[str, CriticallyDampedOrientationState] = {}
    world_quats: dict[str, RotationQuaternion] = {}
    local_quats: dict[str, NDArray[float64]] = {}

    for segment in standard_human.segments:
        ref_geom = reference_geometry.get(segment.name)
        if ref_geom is None:
            continue  # no reference geometry — nothing to solve against
        origin = keypoints.get(segment.origin_keypoint)
        long_pos = keypoints.get(segment.long_axis_keypoint)
        if origin is None or long_pos is None:
            continue  # occluded this frame

        live_vec = np.asarray(long_pos, dtype=np.float64) - np.asarray(
            origin, dtype=np.float64
        )
        norm = float(np.linalg.norm(live_vec))
        if norm < 1e-10:
            continue  # numerically coincident this frame — data, not a declaration error

        live_long = live_vec / norm
        swing = rotation_between_vectors(ref_geom.basis[0], live_long)

        # ── twist: declared keypoint (gated) or damped minimal ─────
        twist_dir: NDArray[float64] | None = None
        if segment.twist_keypoint is not None:
            twist_pos = keypoints.get(segment.twist_keypoint)
            if twist_pos is not None:
                candidate = np.asarray(twist_pos, dtype=np.float64) - np.asarray(
                    origin, dtype=np.float64
                )
                candidate_norm = float(np.linalg.norm(candidate))
                if candidate_norm > 1e-10:
                    candidate = candidate / candidate_norm
                    if (
                        abs(float(np.dot(live_long, candidate)))
                        <= _SINGULARITY_DOT_THRESHOLD
                    ):
                        twist_dir = candidate

        if twist_dir is not None:
            live_basis = build_orthonormal_basis(live_long, twist_dir)
            world_quats[segment.name] = compute_rotation_from_live_basis(
                live_basis, ref_geom.basis
            )
        else:
            approx_live = swing.rotate_vector(ref_geom.basis[1])
            live_basis = build_orthonormal_basis(live_long, approx_live)
            q_raw = compute_rotation_from_live_basis(live_basis, ref_geom.basis)
            previous_state = previous_damping_states.get(segment.name)
            if previous_state is None or timestep_seconds is None:
                damped_state = CriticallyDampedOrientationState.at_rest(q_raw)
            else:
                damped_state = advance_critically_damped_orientation(
                    state=previous_state,
                    target_orientation=q_raw,
                    time_constant_seconds=DEFAULT_TWIST_TIME_CONSTANT_SECONDS,
                    timestep_seconds=timestep_seconds,
                )
            damping_states[segment.name] = damped_state
            world_quats[segment.name] = damped_state.orientation

    # ── local quaternions (D1: q_child_local = conj(q_parent) · q_child) ──
    for segment in standard_human.segments:
        world = world_quats.get(segment.name)
        if world is None:
            continue
        if segment.parent is None or segment.parent not in world_quats:
            local = world
        else:
            local = world_quats[segment.parent].conjugate() * world
        local_quats[segment.name] = np.array(
            [local.w, local.x, local.y, local.z], dtype=np.float64
        )

    world_wxyz = {
        name: np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
        for name, q in world_quats.items()
    }
    return FrameOrientationResult(
        world_quaternions=world_wxyz,
        local_quaternions=local_quats,
        timestamp_seconds=timestamp_seconds,
        damping_states=damping_states,
    )
