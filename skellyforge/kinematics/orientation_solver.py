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

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    build_orthonormal_basis,
    compute_live_bone_basis,
    compute_rotation_from_live_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.quaternion_math import (
    Quaternion,
    hamilton_product,
)
from skellyforge.kinematics.rigid_body_kinematics import (
    _check_strictly_increasing,  # noqa: F401 — used in temporal history
)

if TYPE_CHECKING:
    from numpy import float64
    from skellyforge.skellymodels.standard_human.human_bones import (
        HumanBone,
    )
    from skellyforge.skellymodels.standard_human.standard_human_model import (
        StandardHuman,
    )

logger = logging.getLogger(__name__)

# Threshold for singularity detection: if parent and child bone directions
# are within this angle (radians), the chain-resolved twist is unreliable.
_SINGULARITY_THRESHOLD_RAD = np.deg2rad(5.0)
_SINGULARITY_DOT_THRESHOLD = np.cos(_SINGULARITY_THRESHOLD_RAD)  # ≈ 0.996


# ═══════════════════════════════════════════════════════════════════════
# Per-bone solver (called once per bone per frame)
# ═══════════════════════════════════════════════════════════════════════


def solve_bone_world_orientation(
    bone: "HumanBone",
    live_proximal: NDArray[float64],
    live_distal: NDArray[float64],
    live_twist_direction: NDArray[float64] | None = None,
    previous_world_quaternion: Quaternion | None = None,
) -> Quaternion:
    """Compute the world-frame rotation for a single bone.

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
    previous_world_quaternion : Quaternion or None
        The bone's world quaternion from the previous frame. Used only
        by ``DAMPED_MINIMAL`` for temporal smoothing. ``None`` on the
        first frame or when no history is available.

    Returns
    -------
    Quaternion
        World-frame rotation taking the bone from T-pose to its current
        orientation. Identity means the bone is exactly in T-pose.
    """
    ref_geom = bone.reference_geometry

    # ── Swing rotation (bone long-axis alignment) ────────────────
    ref_bone_vec = ref_geom.bone_vector
    live_bone_vec = live_distal - live_proximal
    live_norm = float(np.linalg.norm(live_bone_vec))
    if live_norm < 1e-10:
        logger.debug(
            "Bone %s has near-zero live length — returning identity.",
            bone.name,
        )
        return Quaternion.identity()
    live_bone_vec = live_bone_vec / live_norm

    swing_quat = rotation_between_vectors(ref_bone_vec, live_bone_vec)

    # ── Twist resolution ────────────────────────────────────────
    policy = bone.twist_policy

    if policy.tier.value == "full_frame":
        # Full-frame: swing-only for now. When ≥3 markers per segment
        # are available, the caller should use solve_bone_full_frame()
        # instead. This path is the fallback when only endpoints exist.
        return swing_quat

    elif policy.tier.value == "chain_resolved":
        return _solve_chain_resolved(
            bone=bone,
            swing_quat=swing_quat,
            live_bone_vec=live_bone_vec,
            live_twist_direction=live_twist_direction,
        )

    elif policy.tier.value == "damped_minimal":
        return _solve_damped_minimal(
            bone=bone,
            swing_quat=swing_quat,
            live_bone_vec=live_bone_vec,
            live_twist_direction=live_twist_direction,
            previous_world_quaternion=previous_world_quaternion,
        )

    else:
        raise ValueError(
            f"Unknown twist tier {policy.tier!r} for bone {bone.name!r}"
        )


def solve_bone_full_frame(
    bone: "HumanBone",
    reference_marker_positions: NDArray[float64],
    live_marker_positions: NDArray[float64],
) -> Quaternion:
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
    Quaternion
    """
    from skellyforge.kinematics.coordinate_frame_ops import (
        align_point_sets_kabsch,
    )

    if len(reference_marker_positions) < 3:
        raise ValueError(
            f"Full-frame bone {bone.name!r} requires >= 3 markers, "
            f"got {len(reference_marker_positions)}"
        )

    R = align_point_sets_kabsch(
        reference_marker_positions, live_marker_positions
    )
    return Quaternion.from_rotation_matrix(R)


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
        ``{bone_name: (4,) wxyz array}`` — parent-relative rotation
        (``hamilton_product(world_child, conjugate(world_parent))``).
        The root bone's local equals its world.
    """

    world_quaternions: dict[str, NDArray[float64]]
    local_quaternions: dict[str, NDArray[float64]]


def solve_frame_orientations(
    standard_human: "StandardHuman",
    live_joint_positions: dict[str, NDArray[float64]],
    *,
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

    Parameters
    ----------
    standard_human : StandardHuman
        The canonical skeleton model.
    live_joint_positions : dict
        ``{bone_name: (3,) position}`` — live joint center positions.
        The key is the bone name; the position is the PROXIMAL joint of
        that bone. The distal joint is the proximal joint of the first
        child bone (or can be looked up from the hierarchy).
    previous_result : FrameOrientationResult or None
        Previous frame's result, for temporal damping.
    child_direction_map : dict or None
        Optional ``{bone_name: child_bone_name}`` mapping that specifies
        which child provides the twist reference for chain-resolved
        bones. If ``None``, the first child in hierarchy order is used.

    Returns
    -------
    FrameOrientationResult
    """
    world_quats: dict[str, Quaternion] = {}
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

        prev_quat = None
        if previous_result is not None:
            prev_wxyz = previous_result.world_quaternions.get(bone.name)
            if prev_wxyz is not None:
                prev_quat = Quaternion(
                    w=float(prev_wxyz[0]),
                    x=float(prev_wxyz[1]),
                    y=float(prev_wxyz[2]),
                    z=float(prev_wxyz[3]),
                )

        world_quat = solve_bone_world_orientation(
            bone=bone,
            live_proximal=proximal,
            live_distal=distal,
            live_twist_direction=twist_dir,
            previous_world_quaternion=prev_quat,
        )
        world_quats[bone.name] = world_quat

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
                # local = world_child * conj(world_parent)
                parent_conj = parent_world.conjugate()
                local = world * parent_conj
                local_quats[bone.name] = np.array(
                    [local.w, local.x, local.y, local.z], dtype=np.float64
                )

    # Convert Quaternion objects to wxyz arrays for the result
    world_wxyz: dict[str, NDArray[float64]] = {}
    for name, q in world_quats.items():
        world_wxyz[name] = np.array(
            [q.w, q.x, q.y, q.z], dtype=np.float64
        )

    return FrameOrientationResult(
        world_quaternions=world_wxyz,
        local_quaternions=local_quats,
    )


# ═══════════════════════════════════════════════════════════════════════
# Twist resolution helpers
# ═══════════════════════════════════════════════════════════════════════


def _solve_chain_resolved(
    bone: "HumanBone",
    swing_quat: Quaternion,
    live_bone_vec: NDArray[float64],
    live_twist_direction: NDArray[float64] | None,
) -> Quaternion:
    """Resolve twist from a child bone's direction."""
    ref_geom = bone.reference_geometry

    # Singularity gate: if the live twist direction is nearly parallel
    # to the bone direction, the cross product is unreliable.
    if live_twist_direction is None:
        # Fall back to damped-minimal with the swing-rotated reference
        # approximate axis
        ref_approx = ref_geom.coordinate_frame.approximate_axis
        live_approx = swing_quat.rotate_vector(ref_approx)
        return _solve_damped_minimal(
            bone=bone,
            swing_quat=swing_quat,
            live_bone_vec=live_bone_vec,
            live_twist_direction=live_approx,
            previous_world_quaternion=None,
        )

    dot = float(np.abs(np.dot(live_bone_vec, live_twist_direction)))
    if dot > _SINGULARITY_DOT_THRESHOLD:
        logger.debug(
            "Bone %s: chain-resolved twist direction is nearly parallel "
            "to bone long axis (|dot|=%.4f > %.4f). Degrading to "
            "damped-minimal.",
            bone.name,
            dot,
            _SINGULARITY_DOT_THRESHOLD,
        )
        ref_approx = bone.reference_geometry.coordinate_frame.approximate_axis
        live_approx = swing_quat.rotate_vector(ref_approx)
        return _solve_damped_minimal(
            bone=bone,
            swing_quat=swing_quat,
            live_bone_vec=live_bone_vec,
            live_twist_direction=live_approx,
            previous_world_quaternion=None,
        )

    # Build live basis from bone direction + twist direction, then
    # compute rotation from reference basis to live basis.
    live_basis = compute_live_bone_basis(live_bone_vec, live_twist_direction)
    ref_basis = ref_geom.coordinate_frame.build_basis_matrix()
    return compute_rotation_from_live_basis(live_basis, ref_basis)


def _solve_damped_minimal(
    bone: "HumanBone",
    swing_quat: Quaternion,
    live_bone_vec: NDArray[float64],
    live_twist_direction: NDArray[float64] | None,
    previous_world_quaternion: Quaternion | None,
) -> Quaternion:
    """Resolve twist with temporal damping toward rest twist.

    If a live twist direction is available (e.g. from the swing-rotated
    reference approximate axis), use it to build the live basis. Then
    SLERP toward the previous frame's quaternion for temporal smoothing.
    """
    ref_geom = bone.reference_geometry
    damping = bone.twist_policy.damping_factor

    if live_twist_direction is None:
        # No twist information at all — rotate reference approximate
        # axis by the swing to get a guess, then damp.
        ref_approx = ref_geom.coordinate_frame.approximate_axis
        live_twist_direction = swing_quat.rotate_vector(ref_approx)

    live_basis = compute_live_bone_basis(live_bone_vec, live_twist_direction)
    ref_basis = ref_geom.coordinate_frame.build_basis_matrix()
    current = compute_rotation_from_live_basis(live_basis, ref_basis)

    if previous_world_quaternion is None:
        return current

    # Temporal damping: SLERP toward previous frame
    return Quaternion.slerp(previous_world_quaternion, current, 1.0 - damping)
