"""A chain: three or more linked segments - a path from a start segment to an end
segment in the tree. Straight (a limb) or branching (the wrist fan). This is the
unit IK/FABRIK solves."""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.skellymodels.standard_human.rigid_body_segment import (
    RigidBodySegment,
)


@dataclass(frozen=True, slots=True)
class KinematicChain:
    name: str
    start_segment: RigidBodySegment
    end_segment: RigidBodySegment

    @property
    def segments(self) -> tuple[RigidBodySegment, ...]:
        """The path from start_segment to end_segment (walk parent edges up from the
        end, then reverse)."""
        path: list[RigidBodySegment] = []
        current: RigidBodySegment | None = self.end_segment
        while current is not None:
            path.append(current)
            if current is self.start_segment:
                break
            current = current.parent
        else:
            raise ValueError(
                f"chain {self.name!r}: end_segment {self.end_segment.name!r} is not "
                f"a descendant of start_segment {self.start_segment.name!r}"
            )
        return tuple(reversed(path))
