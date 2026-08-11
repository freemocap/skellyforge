"""Humanoid bone definitions for the standard human model.

Each bone in the VRM-1.0-aligned humanoid skeleton is a rigid segment with a
reference geometry (T-pose joint centers + a body-fixed coordinate frame) and a
declarative twist policy that tells the orientation solver how to resolve the
underdetermined roll degree of freedom.

Identity quaternion (w=1, x=0, y=0, z=0) means the bone is exactly in its
T-pose reference orientation. This is the contract that every downstream
consumer relies on.

Classes here are `dataclasses` — designed for speed in the per-frame
orientation solve, not for serialization. No aliases live on these objects
(aliases are a serialization concern, handled by `human_bone_aliases.py`).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # numpy arrays in dataclass fields — type-checkers see the annotation
    # but we don't import at runtime
    pass


# ── Twist resolution tiers ────────────────────────────────────────────


class TwistTier(str, Enum):
    """How a bone's twist (roll around its long axis) is resolved.

    Ordered by priority — the solver tries each tier in sequence.
    """

    FULL_FRAME = "full_frame"
    """≥3 non-collinear markers on this segment → full Kabsch orientation."""

    CHAIN_RESOLVED = "chain_resolved"
    """Twist sourced from a child bone's direction (e.g. elbow hinge for upper arm)."""

    DAMPED_MINIMAL = "damped_minimal"
    """Fallback: hold zero/rest twist with critically-damped temporal smoothing."""


# ── Coordinate frame ───────────────────────────────────────────────────


@dataclass
class CoordinateFrameDefinition:
    """A right-handed body-fixed coordinate frame for a single bone.

    Defined by two orthogonal axes:
      - `exact_axis` — the bone's long axis (proximal → distal joint center).
        This is geometrically determined by the skeleton.
      - `approximate_axis` — the twist reference direction. What this IS
        depends on the bone's `TwistPolicy` (child bone direction, marker
        cluster, etc.).
      - The third axis is `normalize(cross(exact, approximate))`, computed on
        demand — never stored independently (it would be redundant and could
        drift from the other two).

    All axes are unit vectors in the canonical coordinate frame at T-pose.
    """

    exact_axis: NDArray[np.float64]  # shape (3,), unit vector
    approximate_axis: NDArray[np.float64]  # shape (3,), unit vector

    def __post_init__(self) -> None:
        """Validate that axes are unit vectors and not parallel."""
        for name in ("exact_axis", "approximate_axis"):
            arr = getattr(self, name)
            if arr.shape != (3,):
                raise ValueError(
                    f"{name} must have shape (3,), got {arr.shape}"
                )
            norm = float(np.linalg.norm(arr))
            if not np.isclose(norm, 1.0, atol=1e-10):
                raise ValueError(
                    f"{name} must be a unit vector, got norm={norm:.6f}"
                )

        # Axes must not be parallel (within ~1° — stiffer than the
        # singularity gate in the orientation solver)
        dot = float(np.abs(np.dot(self.exact_axis, self.approximate_axis)))
        if dot > 0.9998:  # cos(1°) ≈ 0.9998
            raise ValueError(
                f"Exact and approximate axes are nearly parallel "
                f"(|dot|={dot:.6f}). Choose a different approximate axis."
            )

    def compute_third_axis(self) -> NDArray[np.float64]:
        """Compute the third axis via right-handed cross product.

        Returns a new unit vector each call (the stored axes are never
        modified).
        """
        third = np.cross(self.exact_axis, self.approximate_axis)
        norm = float(np.linalg.norm(third))
        if norm < 1e-10:
            raise ValueError("Cross product produced zero vector — axes are parallel")
        return third / norm

    def build_basis_matrix(self) -> NDArray[np.float64]:
        """Build the full (3, 3) orthonormal basis.

        Rows are [exact_axis, approximate_axis, third_axis] in that order.
        """
        basis = np.empty((3, 3), dtype=np.float64)
        basis[0] = self.exact_axis
        basis[1] = self.approximate_axis
        basis[2] = self.compute_third_axis()
        return basis


# ── Reference geometry ─────────────────────────────────────────────────


@dataclass
class BoneReferenceGeometry:
    """T-pose geometry for a single bone.

    Defines where the bone is in space at rest and what its body-fixed
    coordinate frame looks like. The orientation solver compares live
    landmark positions against this reference to compute per-frame
    quaternions.

    Identity quaternion means the bone matches this reference exactly.
    """

    proximal_joint_center: NDArray[np.float64]  # shape (3,), canonical mm
    distal_joint_center: NDArray[np.float64]  # shape (3,), canonical mm
    coordinate_frame: CoordinateFrameDefinition

    @property
    def bone_vector(self) -> NDArray[np.float64]:
        """Unit vector along the bone's long axis (proximal → distal)."""
        vec = self.distal_joint_center - self.proximal_joint_center
        norm = float(np.linalg.norm(vec))
        if norm < 1e-10:
            raise ValueError(
                f"Bone has near-zero length "
                f"(proximal={self.proximal_joint_center}, "
                f"distal={self.distal_joint_center})"
            )
        return vec / norm

    @property
    def bone_length(self) -> float:
        """Euclidean distance between proximal and distal joint centers."""
        return float(np.linalg.norm(
            self.distal_joint_center - self.proximal_joint_center
        ))

    @property
    def mid_point(self) -> NDArray[np.float64]:
        """Midpoint of the bone (center of the segment)."""
        return (self.proximal_joint_center + self.distal_joint_center) / 2.0


# ── Twist policy ───────────────────────────────────────────────────────


@dataclass
class TwistPolicy:
    """How a bone's twist degree of freedom is resolved.

    Encoded declaratively on the bone so the engine reads the policy rather
    than baking per-bone logic into the solver.
    """

    tier: TwistTier
    """Which resolution strategy to use."""

    twist_source_bone: str | None = None
    """For CHAIN_RESOLVED tier: name of the child bone whose direction
    supplies the twist reference. e.g. `left_upper_arm` sources twist from
    `left_lower_arm` (the elbow hinge).

    ``None`` for FULL_FRAME and DAMPED_MINIMAL tiers.
    """

    damping_factor: float = 0.95
    """For DAMPED_MINIMAL tier: critically-damped smoothing factor
    (0 < factor < 1). Higher = more smoothing, slower recovery.

    Default 0.95 is appropriate for 60 Hz capture; tune if framerate
    changes significantly.
    """

    def __post_init__(self) -> None:
        if self.tier == TwistTier.CHAIN_RESOLVED:
            if self.twist_source_bone is None:
                raise ValueError(
                    "CHAIN_RESOLVED twist tier requires twist_source_bone "
                    "to be set (the child bone whose direction supplies "
                    "the twist reference)"
                )
        if self.tier != TwistTier.CHAIN_RESOLVED:
            if self.twist_source_bone is not None:
                raise ValueError(
                    f"twist_source_bone is only meaningful for "
                    f"CHAIN_RESOLVED tier, got tier={self.tier.value}"
                )

        if not 0.0 < self.damping_factor < 1.0:
            raise ValueError(
                f"damping_factor must be in (0, 1), got {self.damping_factor}"
            )


# ── Human bone ─────────────────────────────────────────────────────────


@dataclass
class HumanBone:
    """A single bone in the VRM-1.0-aligned humanoid skeleton.

    Carries everything the orientation solver and downstream adapters need:
    where it is at rest, how its coordinate frame is oriented, how to
    resolve its twist, and where it sits in the hierarchy.

    No aliases. The bone doesn't know it's called ``upperarm_l`` in Unreal.
    That's a serialization concern — see ``human_bone_aliases.py``.
    """

    name: str
    """Canonical snake_case name (e.g. ``left_upper_arm``)."""

    parent: str | None
    """Canonical name of the parent bone, or ``None`` for the root (hips)."""

    required: bool
    """Whether this bone must be present for the skeleton to be valid."""

    reference_geometry: BoneReferenceGeometry
    """T-pose joint centers and coordinate frame."""

    twist_policy: TwistPolicy
    """How to resolve the twist degree of freedom."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("bone name must be non-empty")
        if not self.name.islower():
            raise ValueError(
                f"bone name must be snake_case (all lowercase), "
                f"got '{self.name}'"
            )
        if " " in self.name:
            raise ValueError(
                f"bone name must not contain spaces, got '{self.name}'"
            )
