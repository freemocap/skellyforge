"""The authored unit of the standard human: one rigid-body segment.

A segment is an **origin**, an **orientation**, and a **length**. Its RIGID
POINT SET is declared explicitly (``rigid_points``: every keypoint rigid on the
segment), plus a tuple of tagged **axis declarations** built from that set:

- An EXACT axis — the segment's defining direction, resolved directly from a
  keypoint's position every frame. Its target keypoint must be rigid on the
  segment.
- An APPROXIMATE axis — a *direction reference* for the second basis axis,
  Gram-Schmidt'd against the exact axis. Its target keypoint must also be rigid
  on the segment; a segment with no approximate axis falls to the damped
  minimal-roll tier (the twist-less fallback).

Every axis direction is ``positions[target_keypoint] − positions[origin
keypoint]``. The origin keypoint is the segment's, so an axis direction always
starts at the segment's own origin and ends at another point of the segment's
own rigid geometry: a segment's frame is a function of its own points only.

The tags (``EXACT`` / ``APPROXIMATE``) — not the axis names — carry the roles.

Rationale for the generality: a rigid body is a point set with fixed pairwise
distances. A 2-point segment is the degenerate case — one invariant (its
length and its one axis); 3+ points open the full rigid-body fit (a graded
capacity that lets future trackers enrich any segment by adding points, with
no new concepts).

The T-pose rest fields (``rest_rotation``, ``rest_roll``) stay exactly as
authored — they define the REST frame. The axis declarations define the LIVE
frame: every axis resolves from the segment's own rigid geometry.

Keypoint names are side-agnostic within a part; composition prefixes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ParentAttachment(str, Enum):
    """Which end of the parent this segment's origin sits at."""

    ORIGIN = "origin"
    """Branch from the parent's origin — e.g. both upper legs and the spine
    branch from the hips' origin."""

    DISTAL = "distal"
    """Continue from the parent's distal end — the common case."""


class AxisKind(str, Enum):
    """How a declared axis feeds the frame construction (the role, not the name)."""

    EXACT = "exact"
    """A hard direction from the segment's own rigid geometry; its keypoints
    must be in ``rigid_points``."""

    APPROXIMATE = "approximate"
    """A soft direction reference, Gram-Schmidt-projected; its keypoint must
    be in ``rigid_points``."""


@dataclass(frozen=True)
class AxisDefinition:
    """One declared axis of the segment's local frame (identity == T-pose).

    ``axis`` names WHICH basis vector this declaration defines ("x"/"y"/"z").
    ``kind`` is how the direction feeds the Gram-Schmidt construction: EXACT axes
    are hard directions from the segment's own rigid geometry; APPROXIMATE axes
    are soft direction references (Gram-Schmidt-projected). Both kinds resolve
    their direction as ``positions[target_keypoint] − positions[origin keypoint]``;
    ``target_keypoint`` names the point this axis points toward, and must be a
    member of the segment's ``rigid_points``.
    """

    axis: Literal["x", "y", "z"]
    kind: AxisKind
    target_keypoint: str


@dataclass(frozen=True)
class RotationLimits:
    """Joint range-of-motion limits in degrees, in the segment's local frame.

    **Declared, not yet enforced** — the solver ignores these in SF-SM. They are
    specified now so the model is complete and the values are reviewed alongside
    the anatomy they constrain, rather than bolted on later.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            low, high = getattr(self, axis)
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(
                    f"rotation_limits.{axis} bounds must be finite, got ({low}, {high})"
                )
            if low > high:
                raise ValueError(
                    f"rotation_limits.{axis} lower bound {low} exceeds upper bound {high}"
                )


@dataclass(frozen=True)
class SegmentDefinition:
    """One VRM-1.0-aligned rigid body, authored side-agnostically."""

    name: str
    parent: str | None
    parent_attachment: ParentAttachment
    rigid_points: tuple[str, ...]
    origin_keypoint: str
    axes: tuple[AxisDefinition, ...]
    """Tagged axis declarations, in construction order. **The first axis must be
    EXACT** — it is the segment's long axis (the defining direction), resolved
    first and used to seed the frame. Authored order is construction order:
    indexed ``axes[0]`` is the long axis everywhere. Additional EXACT axes and
    APPROXIMATE direction references follow it. Every axis resolves from the
    segment's own rigid geometry: ``positions[target_keypoint] −
    positions[origin_keypoint]``."""
    rest_rotation: tuple[float, float, float]
    rest_roll: float
    length_ratio: float
    rotation_limits: RotationLimits | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.lower() or " " in self.name:
            raise ValueError(f"segment name must be snake_case, got {self.name!r}")

        rigid = self.rigid_points
        if len(rigid) < 2:
            raise ValueError(
                f"segment {self.name!r}: rigid_points needs at least 2 keypoints "
                f"(a rigid body has at least one defining vector), got {rigid!r}"
            )
        if len(set(rigid)) != len(rigid):
            raise ValueError(
                f"segment {self.name!r}: rigid_points must be distinct, got {rigid!r}"
            )
        if any(not p or not isinstance(p, str) for p in rigid):
            raise ValueError(
                f"segment {self.name!r}: rigid_points entries must be non-empty "
                f"strings, got {rigid!r}"
            )

        if self.origin_keypoint not in rigid:
            raise ValueError(
                f"segment {self.name!r}: origin_keypoint {self.origin_keypoint!r} "
                f"is not in rigid_points {rigid!r}"
            )

        # ── axis-declaration validation (fail-loud) ────────────────────────
        axes = self.axes
        if not 1 <= len(axes) <= 3:
            raise ValueError(
                f"segment {self.name!r}: must declare 1–3 axes, got {len(axes)}"
            )
        names = [a.axis for a in axes]
        if len(set(names)) != len(names):
            raise ValueError(
                f"segment {self.name!r}: axis names must be distinct, got {names!r}"
            )
        if any(n not in ("x", "y", "z") for n in names):
            raise ValueError(
                f"segment {self.name!r}: axis names must be in {{'x','y','z'}}, "
                f"got {names!r}"
            )
        if not any(a.kind is AxisKind.EXACT for a in axes):
            raise ValueError(
                f"segment {self.name!r}: at least one axis must be EXACT, got {axes!r}"
            )
        if axes[0].kind is not AxisKind.EXACT:
            raise ValueError(
                f"segment {self.name!r}: the first declared axis must be EXACT — "
                f"it is the segment's long axis (axes[0]), got {axes[0]!r}"
            )

        for a in axes:
            if not a.target_keypoint or not isinstance(a.target_keypoint, str):
                raise ValueError(
                    f"segment {self.name!r}: axis {a.axis!r} has a non-empty-string "
                    f"target_keypoint requirement, got {a.target_keypoint!r}"
                )
            if a.target_keypoint == self.origin_keypoint:
                raise ValueError(
                    f"segment {self.name!r}: axis {a.axis!r} target_keypoint is "
                    f"the origin keypoint {a.target_keypoint!r}. A segment vector "
                    f"from the origin to itself would be zero-length and no "
                    f"direction could be resolved from it."
                )
            if a.target_keypoint not in rigid:
                raise ValueError(
                    f"segment {self.name!r}: axis {a.axis!r} ({a.kind.value}) "
                    f"target_keypoint {a.target_keypoint!r} is not in rigid_points "
                    f"{rigid!r}. A segment's frame is a function of its own rigid "
                    f"geometry only — every axis target must be rigid on the segment."
                )

        if not math.isfinite(self.length_ratio) or self.length_ratio <= 0.0:
            raise ValueError(
                f"segment {self.name!r}: length_ratio must be finite and > 0, got {self.length_ratio}"
            )

    @property
    def resolves_twist(self) -> bool:
        """Whether this segment's roll is determined by a declared direction.

        ``True`` when an APPROXIMATE direction reference is declared (a second
        basis axis exists → the roll resolves). ``False`` means the twist falls
        back to the critically damped minimal-twist solve — the tier is a
        *consequence* of the declaration, not a separate policy.
        """
        return any(a.kind is AxisKind.APPROXIMATE for a in self.axes)

    def required_keypoints(self) -> set[str]:
        """Every keypoint this segment needs to be solvable.

        The union over every field that references a keypoint by name:
        ``rigid_points ∪ {origin_keypoint} ∪ all axis targets``. Every axis's
        ``target_keypoint`` and the ``origin_keypoint`` are already members of
        ``rigid_points`` (enforced at load), so the set is exactly
        ``rigid_points``.
        """
        names = set(self.rigid_points)
        names.add(self.origin_keypoint)
        for a in self.axes:
            names.add(a.target_keypoint)
        return names
