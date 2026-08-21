from dataclasses import dataclass

from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment

LinkageNameString = str


@dataclass(frozen=True, slots=True)
class SegmentLinkage:
    name: LinkageNameString
    segments: tuple[RigidBodySegment]
