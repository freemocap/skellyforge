"""The ontology's chain layer: declared multi-segment paths over the joints.

A chain is three or more segments connected tip-to-tail by joints - an arm, a
leg, a finger - declared in the skeleton's `chains:` YAML and compiled into
object references here. This is the unit multi-segment math owns: forward
synthesis from joint angles (`core/skeleton/chain/synthesis.py`), inverse
kinematics, twist backfill.

A linear path is the only shape for now; branching fans (the wrist) arrive with
coupled constraints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.linkage.joint_definition import JointDefinition
from skellyforge.type_overloads import ChainNameString, RigidBodySegmentName

MINIMUM_CHAIN_SEGMENTS = 3


@dataclass(frozen=True, slots=True, eq=False)
class KinematicChain:
    """An ordered run of segments joined by joints, proximal first.

    Attributes:
        name: the chain's own name ("left_arm").
        segments: the chain's segments, proximal -> distal (object references).
        joints: the joints BETWEEN consecutive segments, in the same order -
            exactly ``len(segments) - 1`` of them.
    """

    name: ChainNameString
    segments: tuple[RigidBodySegment, ...]
    joints: tuple[JointDefinition, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("chain name must be non-empty")
        if len(self.segments) < MINIMUM_CHAIN_SEGMENTS:
            raise ValueError(
                f"chain {self.name!r}: a chain is {MINIMUM_CHAIN_SEGMENTS}+ "
                f"segments - got {[segment.name for segment in self.segments]}"
            )
        if len(self.joints) != len(self.segments) - 1:
            raise ValueError(
                f"chain {self.name!r}: needs exactly one joint between each pair "
                f"of consecutive segments ({len(self.segments)} segments -> "
                f"{len(self.segments) - 1} joints) - got {len(self.joints)}"
            )

    @classmethod
    def from_yaml_entry(
        cls,
        *,
        name: ChainNameString,
        segment_names: Sequence[RigidBodySegmentName],
        joints: Mapping[RigidBodySegmentName, JointDefinition],
    ) -> "KinematicChain":
        """Compile one `chains:` entry against the skeleton's loaded joints.

        Args:
            name: the chain's name.
            segment_names: the segments, proximal -> distal.
            joints: every joint of the skeleton, keyed by joint name.

        Raises:
            ValueError: a segment name is unknown or repeated, the chain is too
                short, or two consecutive segments share no joint (a broken
                link - the chain would have nothing to synthesize through).
        """
        seen: set[str] = set()
        resolved_segments: list[RigidBodySegment] = []
        for raw_name in segment_names:
            if not isinstance(raw_name, str):
                raise ValueError(
                    f"chain {name!r}: segment entries must be strings - got "
                    f"{raw_name!r}"
                )
            if raw_name in seen:
                raise ValueError(
                    f"chain {name!r}: segment {raw_name!r} appears twice - a chain "
                    f"is a path, not a revisit"
                )
            seen.add(raw_name)
            resolved_segments.append(
                _segment_for_reference(name=name, reference=raw_name, joints=joints)
            )

        chain_joints: list[JointDefinition] = []
        for parent_segment, child_segment in zip(resolved_segments, resolved_segments[1:]):
            connecting = [
                joint
                for joint in joints.values()
                if joint.parent.name == parent_segment.name
                and joint.child.name == child_segment.name
            ]
            if not connecting:
                raise ValueError(
                    f"chain {name!r}: {parent_segment.name!r} and "
                    f"{child_segment.name!r} are consecutive in the chain but no "
                    f"joint joins them - broken link"
                )
            chain_joints.append(connecting[0])

        return cls(name=name, segments=tuple(resolved_segments), joints=tuple(chain_joints))

    def joint_names_in_order(self) -> tuple[str, ...]:
        """The chain's joint names, proximal -> distal."""
        return tuple(joint.name for joint in self.joints)


def _segment_for_reference(
    *,
    name: ChainNameString,
    reference: RigidBodySegmentName,
    joints: Mapping[RigidBodySegmentName, JointDefinition],
) -> RigidBodySegment:
    """One segment object for a chain entry, from whichever joint mentions it."""
    for joint in joints.values():
        if joint.parent.name == reference:
            return joint.parent
        if joint.child.name == reference:
            return joint.child
    raise ValueError(
        f"chain {name!r}: segment {reference!r} is named by no joint of this skeleton"
    )


def _segment_for_reference(
    *,
    name: ChainNameString,
    reference: RigidBodySegmentName,
    joints: Mapping[RigidBodySegmentName, JointDefinition],
) -> RigidBodySegment:
    """One segment object for a chain entry, from whichever joint mentions it."""
    for joint in joints.values():
        if joint.parent.name == reference:
            return joint.parent
        if joint.child.name == reference:
            return joint.child
    raise ValueError(
        f"chain {name!r}: segment {reference!r} is named by no joint of this skeleton"
    )
