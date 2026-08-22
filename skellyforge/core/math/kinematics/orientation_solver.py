"""Per-segment orientation solver: hydrated landmarks -> segment orientations.

Reads a loaded HumanSkeleton + its T-pose reference geometry, and resolves each
segment's world + local quaternions from THIS FRAME's hydrated landmark
positions (identity == T-pose).

A full rigid body (3+ landmarks) solves by a Kabsch fit over its whole landmark
cloud. A 2-landmark segment solves by swing (the primary direction) + twist (the
twist direction, or the critically damped minimal roll when the twist is
unusable). A rigid child inherits its parent's world rotation.

The solver is a pure function: state flows in and out explicitly (SolveState in,
(result, state) out) - no hidden state, no mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skellyforge.type_overloads import FloatArray

from skellyforge.kinematics.coordinate_frame_ops import (
    align_point_sets_kabsch,
    assemble_named_basis,
    axis_index_and_sign,
    compute_rotation_from_live_basis,
    gram_schmidt_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.critically_damped_orientation import (
    CriticallyDampedOrientationState,
    advance_critically_damped_orientation,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton
from skellyforge.skellymodels.standard_human.standard_human_tpose import StandardHumanTPose

_SINGULARITY_DOT_THRESHOLD = np.cos(np.deg2rad(5.0))
DEFAULT_TWIST_TIME_CONSTANT_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class FrameOrientationResult:
    """Per-segment orientations for one frame (result only - no state)."""

    world_quaternions: dict[str, FloatArray]
    local_quaternions: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class OrientationSolveState:
    """The solver's memory across frames (state only - no result)."""

    timestamp_seconds: float = 0.0
    damping_states: dict[str, CriticallyDampedOrientationState] = field(
        default_factory=dict
    )

    @classmethod
    def empty(cls) -> "OrientationSolveState":
        return cls()


@dataclass(frozen=True, slots=True)
class SolveState:
    """One slice per action; future actions (twist, IK) add their own slices."""

    orientation: OrientationSolveState = field(
        default_factory=OrientationSolveState.empty
    )


def _wxyz(q: RotationQuaternion) -> FloatArray:
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


def solve_frame_orientations(
    skeleton: HumanSkeleton,
    tpose: StandardHumanTPose,
    landmarks: dict[str, FloatArray],
    *,
    timestamp_seconds: float,
    state: SolveState,
) -> tuple[FrameOrientationResult, SolveState]:
    """Compute per-segment world + local quaternions for one frame.

    Walks segments in hierarchy order (parents before children), so a parent's
    world quaternion exists when a child composes its local.

    Damping: segments resolved by the damped minimal roll pass through the
    critically damped filter with dt = timestamp_seconds - the previous
    timestamp. First frame (or a non-advancing clock) seeds at rest.
    """
    previous = state.orientation
    timestep: float | None = None
    if previous.timestamp_seconds > 0.0:
        elapsed = timestamp_seconds - previous.timestamp_seconds
        if elapsed > 0.0:
            timestep = elapsed

    damping_states = dict(previous.damping_states)
    world_quats: dict[str, RotationQuaternion] = {}

    for segment in skeleton.segments:
        if segment.rigid_with_parent:
            if segment.parent is not None:
                parent_q = world_quats.get(segment.parent.name)
                if parent_q is not None:
                    world_quats[segment.name] = parent_q
            continue

        geom = tpose.segments.get(segment.name)
        if geom is None:
            continue

        # 3+ landmarks: full rigid body via Kabsch over the whole cloud
        if len(segment.landmarks) >= 3:
            ref_pts: list[FloatArray] = []
            live_pts: list[FloatArray] = []
            for lm in segment.landmarks:
                ref_p = tpose.landmarks.get(lm.name)
                live_p = landmarks.get(lm.name)
                if ref_p is not None and live_p is not None:
                    ref_pts.append(np.asarray(ref_p, dtype=np.float64))
                    live_pts.append(np.asarray(live_p, dtype=np.float64))
            if len(ref_pts) >= 3:
                try:
                    R = align_point_sets_kabsch(
                        np.asarray(ref_pts, dtype=np.float64),
                        np.asarray(live_pts, dtype=np.float64),
                    )
                    world_quats[segment.name] = RotationQuaternion.from_rotation_matrix(R)
                    continue
                except ValueError:
                    pass  # degenerate this frame -> swing + twist

        # 2 landmarks: swing the primary direction, then resolve twist
        origin = landmarks.get(segment.origin_landmark.name)
        if origin is None:
            continue
        primary = segment.primary_axis
        primary_target = landmarks.get(primary.target_landmark)
        if primary_target is None:
            continue
        live_vec = (
            np.asarray(primary_target, dtype=np.float64)
            - np.asarray(origin, dtype=np.float64)
        )
        norm = float(np.linalg.norm(live_vec))
        if norm < 1e-10:
            continue
        primary_idx, primary_sign = axis_index_and_sign(primary.axis)
        live_primary = primary_sign * (live_vec / norm)
        ref_primary = geom.basis[primary_idx]
        swing = rotation_between_vectors(ref_primary, live_primary)

        # twist: the declared twist direction resolves the roll when usable
        if len(segment.axes) >= 2:
            twist = segment.axes[1]
            twist_target = landmarks.get(twist.target_landmark)
            if twist_target is not None:
                twist_vec = (
                    np.asarray(twist_target, dtype=np.float64)
                    - np.asarray(origin, dtype=np.float64)
                )
                twist_norm = float(np.linalg.norm(twist_vec))
                if twist_norm > 1e-10:
                    twist_idx, twist_sign = axis_index_and_sign(twist.axis)
                    live_twist = twist_sign * (twist_vec / twist_norm)
                    if (
                        abs(float(np.dot(live_twist, live_primary)))
                        <= _SINGULARITY_DOT_THRESHOLD
                    ):
                        try:
                            live_basis = gram_schmidt_basis(
                                live_primary, primary_idx, live_twist, twist_idx
                            )
                            world_quats[segment.name] = (
                                compute_rotation_from_live_basis(live_basis, geom.basis)
                            )
                            continue
                        except ValueError:
                            pass  # collinear -> damped minimal roll

        # damped minimal roll: carry the reference second row by the swing
        approx_idx = (primary_idx + 1) % 3
        approx_ref = geom.basis[approx_idx]
        approx_live = swing.rotate_vector(approx_ref)
        live_basis = assemble_named_basis(
            {primary_idx: live_primary, approx_idx: approx_live}
        )
        q_raw = compute_rotation_from_live_basis(live_basis, geom.basis)
        prev_state = damping_states.get(segment.name)
        if prev_state is None or timestep is None:
            damped = CriticallyDampedOrientationState.at_rest(q_raw)
        else:
            damped = advance_critically_damped_orientation(
                state=prev_state,
                target_orientation=q_raw,
                time_constant_seconds=DEFAULT_TWIST_TIME_CONSTANT_SECONDS,
                timestep_seconds=timestep,
            )
        damping_states[segment.name] = damped
        world_quats[segment.name] = damped.orientation

    # local quaternions (q_child_local = conj(q_parent) * q_child)
    local_quats: dict[str, FloatArray] = {}
    for segment in skeleton.segments:
        world = world_quats.get(segment.name)
        if world is None:
            continue
        if segment.parent is None:
            local = world  # the root's local IS its world (convention)
        elif segment.parent.name in world_quats:
            local = world_quats[segment.parent.name].conjugate() * world
        else:
            # The parent did not solve this frame: there is no parent frame to
            # express this segment's local rotation against. Emit no local
            # (occlusion is data) rather than a world rotation mislabelled as a
            # local one, which the renderer would compose incorrectly.
            continue
        local_quats[segment.name] = _wxyz(local)

    world_wxyz = {name: _wxyz(q) for name, q in world_quats.items()}
    result = FrameOrientationResult(
        world_quaternions=world_wxyz, local_quaternions=local_quats
    )
    new_state = SolveState(
        orientation=OrientationSolveState(
            timestamp_seconds=timestamp_seconds, damping_states=damping_states
        )
    )
    return result, new_state
