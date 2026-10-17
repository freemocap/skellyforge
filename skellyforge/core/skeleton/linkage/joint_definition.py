"""The static face of a joint: two segments, the landmark they share, and how
their relative rotation is to be read as named angles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.type_overloads import LinkageNameString

JOINT_TYPES: Final[tuple[str, ...]] = ("ball", "hinge", "universal", "fixed")
"""The joint types a definition may declare. The measured relative orientation is
always the full 3D rotation regardless of type; `type` records the modeling
choice synthesis tasks (forward/inverse kinematics) should respect."""

DEFAULT_JOINT_TYPE: Final[str] = "ball"
DEFAULT_EULER_SEQUENCE: Final[str] = "zyx"


@dataclass(frozen=True, slots=True, eq=False)
class EulerConvention:
    """How a joint's relative orientation is decomposed into named angles.

    Attributes:
        sequence: three letters over x/y/z with no repeated adjacent axis
            (intrinsic order - see ``core/math/kinematics/euler_sequence.py``).
        angle_names: one plain-language name per sequence letter, in order.
        zero_offsets: radians added after decomposition, so a joint whose
            neutral configuration is not the authored T-pose can still report
            clinically-zeroed angles. Defaults to all zeros.
    """

    sequence: str
    angle_names: tuple[str, str, str] = field(default=None)  # type: ignore[assignment]
    zero_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        from skellyforge.core.math.kinematics.euler_sequence import (
            raise_unless_valid_sequence,
        )

        raise_unless_valid_sequence(sequence=self.sequence)
        names = self.angle_names
        if names is None:
            names = tuple(f"{axis}_angle" for axis in self.sequence)
            object.__setattr__(self, "angle_names", names)
        if len(names) != 3 or len(set(names)) != 3:
            raise ValueError(
                f"euler convention {self.sequence!r} needs exactly three distinct "
                f"angle names - got {names}"
            )
        if len(self.zero_offsets) != 3:
            raise ValueError(
                f"euler convention {self.sequence!r} needs three zero offsets - "
                f"got {len(self.zero_offsets)}"
            )

    @classmethod
    def from_sequence(cls, *, sequence: str) -> "EulerConvention":
        """The convention for `sequence` with axes-derived angle names."""
        return cls(sequence=sequence)


@dataclass(frozen=True, slots=True, eq=False)
class JointDefinition:
    """One parent->child edge of the skeleton's topology, plus its convention.

    Attributes:
        name: the joint's own name. Its own namespace - renaming a joint never
            touches topology.
        parent: the proximal segment (object reference, resolved at load).
        child: the distal segment. Its origin sits at `connect_at`.
        connect_at: the PARENT-owned landmark the child's origin sits on.
        joint_type: the modeling DOF choice for synthesis tasks.
        convention: how relative orientations decompose into named angles.
    """

    name: LinkageNameString
    parent: RigidBodySegment
    child: RigidBodySegment
    connect_at: AnatomicalLandmark
    joint_type: str = DEFAULT_JOINT_TYPE
    convention: EulerConvention = field(default_factory=lambda: EulerConvention.from_sequence(sequence=DEFAULT_EULER_SEQUENCE))

    @property
    def angle_names(self) -> tuple[str, str, str]:
        """Qualified scalar names in the convention's decomposition order."""
        first, second, third = self.convention.angle_names
        return f"{self.name}.{first}", f"{self.name}.{second}", f"{self.name}.{third}"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("joint name must be non-empty")
        if self.parent.name == self.child.name:
            raise ValueError(
                f"joint {self.name!r} joins segment {self.parent.name!r} to itself"
            )
        if self.joint_type not in JOINT_TYPES:
            raise ValueError(
                f"joint {self.name!r}: unknown joint_type {self.joint_type!r} - "
                f"known types are {list(JOINT_TYPES)}"
            )
        if self.connect_at.name not in self.parent.landmarks:
            raise ValueError(
                f"joint {self.name!r}: its connect_at landmark "
                f"{self.connect_at.name!r} must be owned by the parent segment "
                f"{self.parent.name!r}, which has {sorted(self.parent.landmarks)}"
            )
