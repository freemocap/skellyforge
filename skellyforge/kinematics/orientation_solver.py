"""Per-segment orientation solver: declared landmarks → segment orientations.

Reads the composed StandardHuman's segment declarations: each segment names
its origin and a tuple of tagged axis declarations (an exact defining
direction, optionally an approximate direction reference), and the solver
resolves its orientation from THIS FRAME's landmark positions against the
T-pose reference geometry (identity == T-pose).

Twist is two-tier, and the declaration IS the policy:
1. A declared APPROXIMATE direction reference, usable this frame (not occluded,
   not within ~5° of the exact axis — the singularity gate) → the roll
   resolves from it.
2. Otherwise → damped minimal roll: the reference approximate axis carried
   by the swing rotation, critically damped (the D3/D4 filter).

The EXACT axis is the segment's defining direction; its name (x/y/z) picks the
reference geometry basis vector the swing maps to the live exact direction.
Every tier keys off the declared axes by name.

A **rigid child** (``rigid_with_parent``) skips the frame build entirely: its
world rotation is the parent's solved world rotation (identity == T-pose, so
the rest local is identity), with no damping of its own. Declared, never
inferred — the model validates the landmark containment at load.

Segments whose landmarks are missing or numerically coincident this frame are
skipped — occlusion is data, and the load-time validation (segment_definition)
makes a *declared* coincidence impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    align_point_sets_kabsch,
    assemble_named_basis,
    axis_index_and_sign,
    build_segment_frame,
    compute_rotation_from_live_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.critically_damped_orientation import (
    CriticallyDampedOrientationState,
    advance_critically_damped_orientation,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisKind,
)

if TYPE_CHECKING:
    from numpy import float64
    from skellyforge.skellymodels.standard_human.reference_geometry import (
        SegmentReferenceGeometry,
    )
    from skellyforge.skellymodels.standard_human.standard_human_model import (
        StandardHuman,
    )

# Twist directions within this angle (radians) of the exact axis cannot build
# a basis — the singularity gate degrades to the damped minimal tier.
_SINGULARITY_THRESHOLD_RAD = np.deg2rad(5.0)
_SINGULARITY_DOT_THRESHOLD = np.cos(_SINGULARITY_THRESHOLD_RAD)  # ≈ 0.996

# The critically damped twist filter's time constant (seconds). Frame-rate
# independent by construction — see critically_damped_orientation.py.
DEFAULT_TWIST_TIME_CONSTANT_SECONDS = 0.05


def _approximate_axis_index(segment, exact_idx: int) -> int:
    """The basis row of the APPROXIMATE axis; the next row after the exact axis
    when twist-less.

    The damped-minimal tier carries the reference approximate vector. A
    twist-less single-axis segment has none, so the reference geometry placed
    the default perpendicular on the next basis row after the exact axis (see
    ``reference_geometry._rest_basis``); the solver carries that same row.
    """
    for a in segment.axes:
        if a.kind is AxisKind.APPROXIMATE:
            return axis_index_and_sign(a.axis)[0]
    return (exact_idx + 1) % 3


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
    landmarks: dict[str, NDArray[float64]],
    *,
    reference_landmarks: dict[str, NDArray[float64]] | None = None,
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
        if segment.rigid_with_parent:
            # Rigid child (declared): no independent solve — inherit the
            # parent's solved world rotation (identity == T-pose, so the rest
            # local is identity). No damping: the parent's rotation is already
            # damped. An unsolved (occluded) parent leaves the child unsolved.
            parent_q = world_quats.get(segment.parent)
            if parent_q is not None:
                world_quats[segment.name] = parent_q
            continue

        ref_geom = reference_geometry.get(segment.name)
        if ref_geom is None:
            continue  # no reference geometry — nothing to solve against

        # Full rigid bodies (≥3 declared landmarks) solve their rotation by a
        # Kabsch fit over the WHOLE landmark cloud — the skull (head), hips, feet
        # and toes are rigid bodies whose orientation is over-determined by their
        # pairwise geometry, not 2-point spans. Falls back to swing+twist when the
        # reference cloud is absent or too few points hydrate this frame.
        if reference_landmarks is not None and len(segment.landmarks) >= 3:
            ref_pts: list[NDArray[float64]] = []
            live_pts: list[NDArray[float64]] = []
            for name in segment.landmarks:
                ref_p = reference_landmarks.get(name)
                live_p = landmarks.get(name)
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
                    pass  # collinear/degenerate this frame — fall through to swing+twist

        origin = landmarks.get(segment.origin_landmark)
        if origin is None:
            continue  # occluded this frame

        live_vec = None
        if segment.axes:
            # The EXACT axis is the segment's defining direction, declared on
            # whichever local axis (x/y/z) the author chose — no positional
            # read. The solver resolves the same direction build_segment_frame
            # derives from the EXACT declaration: origin → target_landmark.
            exact_axis = segment.exact_axis
            exact_to = landmarks.get(exact_axis.target_landmark)
            if origin is not None and exact_to is not None:
                live_vec = np.asarray(exact_to, dtype=np.float64) - np.asarray(
                    origin, dtype=np.float64
                )
        if live_vec is None:
            continue  # occluded this frame (no usable exact-axis landmarks)
        norm = float(np.linalg.norm(live_vec))
        if norm < 1e-10:
            continue  # numerically coincident this frame — data, not a declaration error

        exact_idx, exact_sign = axis_index_and_sign(exact_axis.axis)
        live_exact = exact_sign * (live_vec / norm)
        # The swing aligns the reference geometry's basis row NAMED by the
        # EXACT axis onto the live exact direction — never a positional read.
        ref_exact_vector = ref_geom.basis[exact_idx]
        swing = rotation_between_vectors(ref_exact_vector, live_exact)

        # ── twist: declared direction reference (gated) or damped minimal ─────
        # ``build_segment_frame`` builds the LIVE frame from the tagged axis
        # declarations directly from this frame's landmark positions. Its
        # internal singularity gate (collinearity_threshold) reproduces the
        # solver's ~5° gate: the APPROXIMATE axis resolves the roll only when
        # non-collinear.
        live_basis, resolved = build_segment_frame(
            segment.axes,
            landmarks,
            segment.origin_landmark,
            collinearity_threshold=_SINGULARITY_DOT_THRESHOLD,
        )

        if resolved:
            world_quats[segment.name] = compute_rotation_from_live_basis(
                live_basis, ref_geom.basis
            )
        else:
            # The damped-minimal tier carries the reference approximate vector —
            # the basis vector NAMED by the APPROXIMATE axis (or the next row
            # after the exact axis for a twist-less segment) — rotated by the
            # swing, then reassembles the frame on the same named rows as the
            # reference geometry.
            approx_idx = _approximate_axis_index(segment, exact_idx)
            approx_ref = ref_geom.basis[approx_idx]
            approx_live = swing.rotate_vector(approx_ref)
            live_basis = assemble_named_basis(
                {exact_idx: live_exact, approx_idx: approx_live}
            )
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
