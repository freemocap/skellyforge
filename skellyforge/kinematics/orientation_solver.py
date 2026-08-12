"""Per-bone orientation solver for the standard human model.

The solver that ties SF-SH-1 (bone model), SF-SH-3 (kinematics math), and
live skeleton positions together: for each frame, walks the bone hierarchy
root-first and produces per-bone world + local quaternions satisfying the
identity == T-pose contract.

Architecture
------------
The solver is a pure function of (model, live positions, optional previous
frame state). It does not import from FreeMoCap or access any pipeline.
The caller (the freemocap aggregator, SF-SH-5) is responsible for feeding
it live joint positions each frame.

Twist resolution
----------------
Each bone's ``TwistPolicy`` determines how the underconstrained roll
degree of freedom is resolved. The solver dispatches:

1. **Full-frame** (≥3 non-collinear markers on the segment)
   → ``align_point_sets_kabsch`` between reference and live marker
   positions. Used for head, pelvis, thorax, hands, feet.

2. **Chain-resolved** (twist from child bone direction)
   → The child bone's live direction supplies the approximate axis.
   ``build_orthonormal_basis`` + ``compute_rotation_from_live_basis``.
   Used for upper arm (elbow hinge), upper leg (knee hinge).

3. **Damped-minimal** (fallback when twist source is occluded)
   → Reference approximate axis rotated by the swing rotation, then
   temporally damped. Used when chain-resolved source is unavailable.

Singularity gate: when parent and child bone directions are within ~5° of
parallel (arm fully straight), chain-resolved degrades to damped-minimal
to avoid the cross-product blowup.

References
----------
- Kabsch (1976), Umeyama (1991) — point-set alignment (full-frame bones).
- Shoemake (1985) — SLERP for temporal damping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    align_point_sets_kabsch,
    compute_live_bone_basis,
    compute_rotation_from_live_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.critically_damped_orientation import (
    CriticallyDampedOrientationState,
    advance_critically_damped_orientation,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.human_bones import TwistTier

if TYPE_CHECKING:
    from numpy import float64
    from skellyforge.skellymodels.standard_human.human_bones import (
        HumanBone,
    )
    from skellyforge.skellymodels.standard_human.standard_human_model import (
        StandardHuman,
    )

# Threshold for singularity detection: if parent and child bone directions
# are within this angle (radians), the chain-resolved twist is unreliable.
_SINGULARITY_THRESHOLD_RAD = np.deg2rad(5.0)
_SINGULARITY_DOT_THRESHOLD = np.cos(_SINGULARITY_THRESHOLD_RAD)  # ≈ 0.996


# ═══════════════════════════════════════════════════════════════════════
# Per-bone solver (called once per bone per frame)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BoneOrientationSolution:
    """One segment's raw (undamped) orientation, plus which tier resolved it.

    ``twist_resolved_by_damped_minimal`` is ``True`` when the segment's twist was
    resolved by the fallback tier — either because its policy declares it, or
    because a ``CHAIN_RESOLVED`` segment degraded (twist source missing, or the
    singularity gate tripped). It tells the frame-level solver which segments to
    run through the critically damped filter.

    Damping is deliberately **not** applied here: it is a function of elapsed time,
    and a single-segment call has no ``dt``. See
    :func:`solve_frame_orientations`.
    """

    orientation: RotationQuaternion
    twist_resolved_by_damped_minimal: bool


def solve_bone_world_orientation(
    bone: "HumanBone",
    live_proximal: NDArray[float64],
    live_distal: NDArray[float64],
    live_twist_direction: NDArray[float64] | None = None,
) -> BoneOrientationSolution:
    """Compute the raw world-frame rotation for a single bone.

    Parameters
    ----------
    bone : HumanBone
        Bone definition with reference geometry and twist policy.
    live_proximal : (3,)
        Live position of the bone's proximal joint (mm, canonical frame).
    live_distal : (3,)
        Live position of the bone's distal joint (mm, canonical frame).
    live_twist_direction : (3,) or None
        Unit vector for the twist-reference direction in the live
        configuration. Required for ``CHAIN_RESOLVED`` tier; optional
        for ``FULL_FRAME`` (not used); used as the approximate axis for
        ``DAMPED_MINIMAL`` if provided, else the reference approximate
        axis rotated by swing is used.

    Returns
    -------
    BoneOrientationSolution
        The **undamped** world-frame rotation taking the bone from T-pose to its
        current orientation (identity means exactly T-pose), and whether the
        damped-minimal tier resolved the twist.
    """
    ref_geom = bone.reference_geometry

    # ── Swing rotation (bone long-axis alignment) ────────────────
    ref_bone_vec = ref_geom.bone_vector
    live_bone_vec = live_distal - live_proximal
    live_norm = float(np.linalg.norm(live_bone_vec))
    if live_norm < 1e-10:
        raise ValueError(
            f"Bone {bone.name!r} has coincident live proximal and distal joints "
            f"({live_proximal} and {live_distal}); its direction is undefined, so "
            f"no orientation can be resolved."
        )
    live_bone_vec = live_bone_vec / live_norm

    swing_quat = rotation_between_vectors(ref_bone_vec, live_bone_vec)

    # ── Twist resolution ────────────────────────────────────────
    tier = bone.twist_policy.tier

    if tier == TwistTier.FULL_FRAME:
        # Full-frame: swing-only for now. When >=3 markers per segment
        # are available, the caller should use solve_bone_full_frame()
        # instead. This path is the fallback when only endpoints exist.
        return BoneOrientationSolution(
            orientation=swing_quat,
            twist_resolved_by_damped_minimal=False,
        )

    if tier == TwistTier.CHAIN_RESOLVED:
        return _solve_chain_resolved(
            bone=bone,
            swing_quat=swing_quat,
            live_bone_vec=live_bone_vec,
            live_twist_direction=live_twist_direction,
        )

    if tier == TwistTier.DAMPED_MINIMAL:
        return BoneOrientationSolution(
            orientation=_solve_minimal_twist(
                bone=bone,
                swing_quat=swing_quat,
                live_bone_vec=live_bone_vec,
                live_twist_direction=live_twist_direction,
            ),
            twist_resolved_by_damped_minimal=True,
        )

    raise ValueError(
            f"Unknown twist tier {bone.twist_policy.tier!r} for bone {bone.name!r}"
        )


def solve_bone_full_frame(
    bone: "HumanBone",
    reference_marker_positions: NDArray[float64],
    live_marker_positions: NDArray[float64],
) -> RotationQuaternion:
    """Compute world rotation for a full-frame bone via Kabsch alignment.

    Parameters
    ----------
    bone : HumanBone
    reference_marker_positions : (M, 3)
        Marker positions in the reference (T-pose) configuration.
    live_marker_positions : (M, 3)
        Corresponding marker positions in the live configuration.

    Returns
    -------
    RotationQuaternion
    """
    if len(reference_marker_positions) < 3:
        raise ValueError(
            f"Full-frame bone {bone.name!r} requires >= 3 markers, "
            f"got {len(reference_marker_positions)}"
        )

    R = align_point_sets_kabsch(
        reference_marker_positions, live_marker_positions
    )
    return RotationQuaternion.from_rotation_matrix(R)


# ═══════════════════════════════════════════════════════════════════════
# Full-skeleton solver (one call per frame)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FrameOrientationResult:
    """Orientations for every bone in the skeleton at one frame.

    Parameters
    ----------
    world_quaternions : dict
        ``{bone_name: (4,) wxyz array}`` — world-frame rotation from
        T-pose for each bone.
    local_quaternions : dict
        ``{bone_name: (4,) wxyz array}`` — parent-relative rotation,
        ``conjugate(world_parent) * world_child``. The root bone's local
        equals its world.
    timestamp_seconds : float
        When this frame was solved. The next frame's ``dt`` is measured
        against it, so the damping filter is driven by real elapsed time
        rather than a frame count.
    damping_states : dict
        ``{bone_name: CriticallyDampedOrientationState}`` for the segments
        whose twist was resolved by the damped-minimal tier. Carried into the
        next frame; segments that did not need damping are absent.

        Held **on the result**, not in module scope, so two pipelines in one
        process cannot contaminate each other's smoothing and a new session
        starts clean.
    """

    world_quaternions: dict[str, NDArray[float64]]
    local_quaternions: dict[str, NDArray[float64]]
    timestamp_seconds: float
    damping_states: dict[str, CriticallyDampedOrientationState] = field(
        default_factory=dict
    )


def solve_frame_orientations(
    standard_human: "StandardHuman",
    live_joint_positions: dict[str, NDArray[float64]],
    *,
    timestamp_seconds: float,
    previous_result: FrameOrientationResult | None = None,
    child_direction_map: dict[str, str] | None = None,
) -> FrameOrientationResult:
    """Compute per-bone quaternions for all bones at one frame.

    Walks the bone hierarchy root-first so that parent world quaternions
    are available when computing child local quaternions. For
    ``CHAIN_RESOLVED`` bones, uses the *child* bone's live direction as
    the twist reference — so those children must already be solved.

    Strategy: two passes.
    1. **Swing-only pass** — compute the swing rotation for every bone
       from its live endpoints. This gives us provisional world quats
       for all bones.
    2. **Twist pass** — for chain-resolved bones, use the (already
       swing-computed) child bone direction as the twist reference and
       recompute the full orientation.

    Damping
    -------
    Segments whose twist was resolved by the damped-minimal tier — whether by
    policy or by a ``CHAIN_RESOLVED`` segment degrading — are passed through the
    critically damped filter before their world quaternion is recorded, using
    ``dt = timestamp_seconds - previous_result.timestamp_seconds``. Every
    fallback path damps; that is the case damping exists for.

    Parameters
    ----------
    standard_human : StandardHuman
        The canonical skeleton model.
    live_joint_positions : dict
        ``{bone_name: (3,) position}`` — live joint center positions.
        The key is the bone name; the position is the PROXIMAL joint of
        that bone. The distal joint is the proximal joint of the first
        child bone (or can be looked up from the hierarchy).
    timestamp_seconds : float
        This frame's time. Damping is driven by real elapsed time, so this must
        advance between frames for smoothing to apply. **Required, deliberately
        without a default** — a default would let a caller silently disable
        damping by omission, which is the kind of quiet degradation that is far
        harder to notice than a missing argument.
    previous_result : FrameOrientationResult or None
        Previous frame's result, carrying the per-segment damping state and the
        timestamp ``dt`` is measured from. ``None`` on the first frame, where
        every damped segment is seeded at its raw orientation with zero velocity.
    child_direction_map : dict or None
        Optional ``{bone_name: child_bone_name}`` mapping that specifies
        which child provides the twist reference for chain-resolved
        bones. If ``None``, the first child in hierarchy order is used.

    Returns
    -------
    FrameOrientationResult
    """
    timestep_seconds: float | None = None
    if previous_result is not None:
        elapsed = timestamp_seconds - previous_result.timestamp_seconds
        # A non-advancing clock cannot drive a time-based filter. Rather than
        # fabricate a dt, treat the frame as a fresh start: each damped segment
        # re-seeds at its raw orientation. Long gaps need no special case — the
        # filter's exponential decay lands them on target with zero velocity.
        timestep_seconds = elapsed if elapsed > 0.0 else None

    previous_damping_states: dict[str, CriticallyDampedOrientationState] = (
        previous_result.damping_states if previous_result is not None else {}
    )
    damping_states: dict[str, CriticallyDampedOrientationState] = {}
    world_quats: dict[str, RotationQuaternion] = {}
    local_quats: dict[str, NDArray[float64]] = {}

    # Build bone lookup
    bone_map: dict[str, "HumanBone"] = {
        b.name: b for b in standard_human.bones
    }

    # Build a joint-position lookup that gives us proximal AND distal
    # for each bone. The distal joint of bone A is the proximal joint
    # of a child of A.
    def _get_distal_position(bone: "HumanBone") -> NDArray[float64] | None:
        # First child's proximal joint = this bone's distal joint
        children = standard_human.get_children(bone.name)
        for child in children:
            pos = live_joint_positions.get(child.name)
            if pos is not None:
                return pos
        # Leaf bone with no tracked child — the distal must be provided
        # directly in live_joint_positions keyed by a synthetic key or
        # by the bone's own name concatenated (e.g. 'head_distal').
        # This is handled by the tracker-to-canonical mapping (ST-SH-2).
        return None

    # ── Pass 1: swing-only world quaternions for all bones ───────
    # We walk in declaration order (which should be hierarchy order).
    bones_to_solve = list(standard_human.bones)

    for bone in bones_to_solve:
        proximal = live_joint_positions.get(bone.name)
        if proximal is None:
            # Bone not present in this frame's data — skip
            continue

        distal = _get_distal_position(bone)
        if distal is None:
            # Leaf bone with no tracked child joint — skip
            continue

        # Get twist direction for chain-resolved bones from child
        twist_dir = None
        if bone.twist_policy.tier.value == "chain_resolved":
            twist_source_name = bone.twist_policy.twist_source_bone
            if twist_source_name and twist_source_name in live_joint_positions:
                # Compute child bone direction from live positions
                source_bone = bone_map.get(twist_source_name)
                if source_bone:
                    source_proximal = live_joint_positions.get(
                        twist_source_name
                    )
                    source_distal = _get_distal_position(source_bone)
                    if (
                        source_proximal is not None
                        and source_distal is not None
                    ):
                        twist_vec = source_distal - source_proximal
                        twist_norm = float(np.linalg.norm(twist_vec))
                        if twist_norm > 1e-10:
                            twist_dir = twist_vec / twist_norm

        solution = solve_bone_world_orientation(
            bone=bone,
            live_proximal=proximal,
            live_distal=distal,
            live_twist_direction=twist_dir,
        )

        if not solution.twist_resolved_by_damped_minimal:
            world_quats[bone.name] = solution.orientation
            continue

        # ── Critically damped twist smoothing ────────────────────
        previous_state = previous_damping_states.get(bone.name)
        if previous_state is None or timestep_seconds is None:
            # First frame for this segment, or a clock that did not advance:
            # seed the filter at the raw orientation with zero velocity.
            damped_state = CriticallyDampedOrientationState.at_rest(
                solution.orientation
            )
        else:
            damped_state = advance_critically_damped_orientation(
                state=previous_state,
                target_orientation=solution.orientation,
                time_constant_seconds=bone.twist_policy.twist_time_constant_seconds,
                timestep_seconds=timestep_seconds,
            )

        damping_states[bone.name] = damped_state
        world_quats[bone.name] = damped_state.orientation

    # ── Compute local quaternions ────────────────────────────────
    for bone in bones_to_solve:
        world = world_quats.get(bone.name)
        if world is None:
            continue

        if bone.parent is None:
            # Root: local = world
            local_quats[bone.name] = np.array(
                [world.w, world.x, world.y, world.z], dtype=np.float64
            )
        else:
            parent_world = world_quats.get(bone.parent)
            if parent_world is None:
                # Parent not solved — fall back to world
                local_quats[bone.name] = np.array(
                    [world.w, world.x, world.y, world.z], dtype=np.float64
                )
            else:
                # A world quaternion maps segment-frame -> world, so world
                # composition is q_child_world = q_parent_world * q_child_local.
                # Inverting gives the parent-relative rotation:
                #     q_child_local = conj(q_parent_world) * q_child_world
                # Operand order is load-bearing — the reverse yields the correct
                # angle about the wrong axis (the delta in world frame rather than
                # relative to the parent).
                local = parent_world.conjugate() * world
                local_quats[bone.name] = np.array(
                    [local.w, local.x, local.y, local.z], dtype=np.float64
                )

    # Convert RotationQuaternion objects to wxyz arrays for the result
    world_wxyz: dict[str, NDArray[float64]] = {}
    for name, q in world_quats.items():
        world_wxyz[name] = np.array(
            [q.w, q.x, q.y, q.z], dtype=np.float64
        )

    return FrameOrientationResult(
        world_quaternions=world_wxyz,
        local_quaternions=local_quats,
        timestamp_seconds=timestamp_seconds,
        damping_states=damping_states,
    )


# ═══════════════════════════════════════════════════════════════════════
# Twist resolution helpers
# ═══════════════════════════════════════════════════════════════════════


def _solve_chain_resolved(
    bone: "HumanBone",
    swing_quat: RotationQuaternion,
    live_bone_vec: NDArray[float64],
    live_twist_direction: NDArray[float64] | None,
) -> BoneOrientationSolution:
    """Resolve twist from a child bone's direction, degrading when it cannot.

    Two conditions force the fallback: no twist source this frame (occlusion), and
    the **singularity gate** — a twist direction within ~5 degrees of the bone's own
    long axis, where the cross product that builds the basis is numerically
    worthless.

    Both return ``twist_resolved_by_damped_minimal=True`` so the frame-level solver
    damps them. Skipping damping here is what made the fallback pop in exactly the
    situation damping exists for.
    """
    ref_geom = bone.reference_geometry

    twist_source_is_unusable = live_twist_direction is None or (
        float(np.abs(np.dot(live_bone_vec, live_twist_direction)))
        > _SINGULARITY_DOT_THRESHOLD
    )

    if twist_source_is_unusable:
        return BoneOrientationSolution(
            orientation=_solve_minimal_twist(
                bone=bone,
                swing_quat=swing_quat,
                live_bone_vec=live_bone_vec,
                live_twist_direction=None,
            ),
            twist_resolved_by_damped_minimal=True,
        )

    # Build live basis from bone direction + twist direction, then
    # compute rotation from reference basis to live basis.
    live_basis = compute_live_bone_basis(live_bone_vec, live_twist_direction)
    ref_basis = ref_geom.coordinate_frame.build_basis_matrix()
    return BoneOrientationSolution(
        orientation=compute_rotation_from_live_basis(live_basis, ref_basis),
        twist_resolved_by_damped_minimal=False,
    )


def _solve_minimal_twist(
    bone: "HumanBone",
    swing_quat: RotationQuaternion,
    live_bone_vec: NDArray[float64],
    live_twist_direction: NDArray[float64] | None,
) -> RotationQuaternion:
    """Resolve twist by holding the rest twist — the minimal-twist estimate.

    With no usable twist source, the reference approximate axis is carried into the
    live configuration by the swing rotation. That is the "no roll beyond what the
    swing implies" answer, and it is inherently noisy frame to frame, which is why
    the frame-level solver runs this tier's output through the critically damped
    filter (:mod:`skellyforge.kinematics.critically_damped_orientation`).

    This function is **undamped** — damping needs elapsed time, which a per-segment
    call does not have.
    """
    ref_geom = bone.reference_geometry

    if live_twist_direction is None:
        live_twist_direction = swing_quat.rotate_vector(
            ref_geom.coordinate_frame.approximate_axis
        )

    live_basis = compute_live_bone_basis(live_bone_vec, live_twist_direction)
    ref_basis = ref_geom.coordinate_frame.build_basis_matrix()
    return compute_rotation_from_live_basis(live_basis, ref_basis)
