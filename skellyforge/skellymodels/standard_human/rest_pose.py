"""The standard-human rest pose: frozen-slots projections of the authored
segments + reference geometry into the resolved T-pose that rides the wire.

These are the "third thing" that completes the ontology: SegmentDefinition is
the authored declaration, SegmentReferenceGeometry is the per-segment resolved
geometry, and RestSegment is the resolved segment (the two folded together).
They are pure data (frozen slots) plus a to_cbor_message() that returns plain
Python types - no cbor2 import here; the freemocap streaming layer owns the
actual CBOR encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SIGNED_AXES = ("x", "y", "z", "-x", "-y", "-z")


@dataclass(frozen=True, slots=True)
class LongitudinalAxis:
    """The bone's longitudinal direction in its local rest frame: a signed
    basis axis name ("x"/"y"/"z"/"-x"/"-y"/"-z") or a normalized 3-vector.
    The longitudinal axis is the vector from the segment origin to its child
    (or to its tip for a leaf segment with no child)."""

    value: str | tuple[float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.value, str):
            if self.value not in _SIGNED_AXES:
                raise ValueError(f"unknown longitudinal axis {self.value!r}")
        else:
            vec = self.value
            if len(vec) != 3:
                raise ValueError("longitudinal direction must be a 3-tuple")
            norm = sum(float(c) * float(c) for c in vec) ** 0.5
            if norm <= 0.0:
                raise ValueError("longitudinal direction must be non-zero")
            object.__setattr__(self, "value", tuple(float(c) / norm for c in vec))

    @classmethod
    def from_axis(cls, axis: str) -> "LongitudinalAxis":
        return cls(value=axis)

    @classmethod
    def from_direction(cls, direction) -> "LongitudinalAxis":
        return cls(value=tuple(float(c) for c in direction))

    def to_cbor_message(self) -> str | list[float]:
        return self.value if isinstance(self.value, str) else list(self.value)


@dataclass(frozen=True, slots=True)
class RestSegment:
    """One RESOLVED segment, folded from SegmentDefinition + SegmentReferenceGeometry.
    Carries resolved T-pose facts only, never a construction input."""

    name: str
    parent: str | None
    longitudinal_axis: LongitudinalAxis
    rest_orientation: tuple[float, float, float, float]  # wxyz
    length_mm: float
    rigid_with_parent: bool = False

    @classmethod
    def from_geometry(cls, segment, geometry) -> "RestSegment":
        from skellyforge.kinematics.quaternion_math import RotationQuaternion  # noqa: PLC0415
        q = RotationQuaternion.from_rotation_matrix(geometry.basis.T)
        return cls(
            name=segment.name,
            parent=segment.parent,
            longitudinal_axis=LongitudinalAxis.from_axis(segment.exact_axis.axis),
            rest_orientation=(float(q.w), float(q.x), float(q.y), float(q.z)),
            length_mm=float(geometry.length),
            rigid_with_parent=segment.rigid_with_parent,
        )

    def to_cbor_message(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "longitudinal_axis": self.longitudinal_axis.to_cbor_message(),
            "rest_orientation": list(self.rest_orientation),
            "length_mm": self.length_mm,
            "rigid_with_parent": self.rigid_with_parent,
        }


@dataclass(frozen=True, slots=True)
class RestLandmark:
    """One RESOLVED landmark: its name and its rest position."""

    name: str
    rest_position: tuple[float, float, float]

    @classmethod
    def from_position(cls, name: str, position) -> "RestLandmark":
        return cls(name=name, rest_position=(float(position[0]), float(position[1]), float(position[2])))

    def to_cbor_message(self) -> dict[str, Any]:
        return {"name": self.name, "rest_position": list(self.rest_position)}
