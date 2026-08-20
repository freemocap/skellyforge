"""A linkage: two segments that share a point (e.g. upper arm + lower arm at the
elbow). Derived from the parent edges - the child's origin_landmark IS the
shared point."""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.skellymodels.standard_human.anatomical_landmark import (
    AnatomicalLandmark,
)
from skellyforge.skellymodels.standard_human.rigid_body_segment import (
    RigidBodySegment,
)

JointNameString = str

@dataclass(frozen=True, slots=True)
class JointLinkage:
    name: JointNameString
    parent_segment: RigidBodySegment
    child_segment: RigidBodySegment
    shared_landmark: AnatomicalLandmark
