"""The authored unit of the standard human: one rigid-body segment.

A segment is an **origin**, an **orientation**, and a **length**. Its RIGID
POINT SET is declared explicitly (``landmarks``: every landmark rigid on the
segment), plus a tuple of tagged **axis declarations** built from that set:

- An EXACT axis — the segment's defining direction, resolved directly from a
  landmark's position every frame, on whichever local axis (x/y/z) the author
  declares. Its target landmark must be rigid on the segment.
- An APPROXIMATE axis — a *direction reference* for a second basis axis,
  Gram-Schmidt'd against the exact axis. Its target landmark must also be rigid
  on the segment; a segment with no approximate axis falls to the damped
  minimal-roll tier (the twist-less fallback).

Every axis direction is ``positions[target_landmark] − positions[origin
landmark]``. The origin landmark is the segment's, so an axis direction always
starts at the segment's own origin and ends at another point of the segment's
own rigid geometry: a segment's frame is a function of its own points only.

The tags (``EXACT`` / ``APPROXIMATE``) — not the axis names — carry the roles.
The axis NAME (x/y/z) names which basis vector a declaration defines. The
construction machinery derives everything from the names and kinds; it makes no
assumption about which slot in the ``axes`` tuple holds the exact axis (a
segment may declare its exact axis on x, y, or z) nor about the tuple order —
construction follows basis order (x, y, z), not the authored tuple order.

Rationale for the generality: a rigid body is a point set with fixed pairwise
distances. A 2-point segment is the degenerate case — one invariant (its
length and its one axis); 3+ points open the full rigid-body fit (a graded
capacity that lets future trackers enrich any segment by adding points, with
no new concepts).

The T-pose rest field (``rest_rotation``) stays exactly as authored — it
defines the REST frame. The axis declarations define the LIVE frame: every
axis resolves from the segment's own rigid geometry.

Authoring convention (VRM 1.0 local frame, documented once here):

Every segment's rest frame is right-handed. The body/hand segments declare
their exact axis on **y** with **+Y toward the child bone** (the VRM 1.0
humanoid rule); the face bones declare their exact axis on **z** with **+Z =
the gaze direction** (VRM's face-bone rule). The remaining basis axes are not a
single fixed world direction: a declared second/third point gives the segment
its own reference direction (Gram-Schmidt'd against the exact axis), and where
none is declared the rest frame follows that segment's T-pose geometry — the
third axis derives from the per-segment declarations plus the T-pose geometry,
not a global "+X". Each segment's ``rest_rotation`` extrudes the declared axis
name through its euler triple so that the rest frame's named vector equals that
authored direction.

Landmark names are side-agnostic within a part; composition prefixes them.
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
    """A hard direction from the segment's own rigid geometry; its landmarks
    must be in ``landmarks``."""

    APPROXIMATE = "approximate"
    """A soft direction reference, Gram-Schmidt-projected; its landmark must
    be in ``landmarks``."""


@dataclass(frozen=True)
class AxisDefinition:
    """One declared axis of the segment's local frame (identity == T-pose).

    ``axis`` names WHICH basis vector this declaration defines ("x"/"y"/"z");
    it carries no positional meaning — the author may declare the exact axis on
    any of x/y/z, and the construction machinery reads the names. ``kind`` is
    how the direction feeds the Gram-Schmidt construction: EXACT axes are hard
    directions from the segment's own rigid geometry; APPROXIMATE axes are soft
    direction references (Gram-Schmidt-projected). Both kinds resolve their
    direction as ``positions[target_landmark] − positions[origin landmark]``;
    ``target_landmark`` names the point this axis points toward, and must be a
    member of the segment's ``landmarks``.
    """

    axis: Literal["x", "y", "z"]
    kind: AxisKind
    target_landmark: str


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
    landmarks: tuple[str, ...]
    origin_landmark: str
    axes: tuple[AxisDefinition, ...]
    """Tagged axis declarations, in construction order. The EXACT axis is the
    segment's defining direction, declared on whichever local axis the author
    chooses; construction order is the basis order (x, y, z), not this tuple's
    order. At least one axis (any position) must be EXACT; additional EXACT
    axes and APPROXIMATE direction references follow by name. Every axis
    resolves from the segment's own rigid geometry: ``positions[target_landmark]
    − positions[origin_landmark]``."""
    rest_rotation: tuple[float, float, float]
    length_ratio: float
    rotation_limits: RotationLimits | None = None
    rigid_with_parent: bool = False
    """Rigid child (declared, never inferred): the segment's pose is not solved
    from its own hydrated landmarks — it inherits the parent's solved pose
    composed with its rest local rotation (identity == T-pose). Requires a
    parent; the composed model additionally validates that every landmark is a
    member of the parent's landmark set (a rigid child's geometry must live
    inside its parent's rigid set)."""

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.lower() or " " in self.name:
            raise ValueError(f"segment name must be snake_case, got {self.name!r}")

        rigid = self.landmarks
        if len(rigid) < 2:
            raise ValueError(
                f"segment {self.name!r}: landmarks needs at least 2 landmarks "
                f"(a rigid body has at least one defining vector), got {rigid!r}"
            )
        if len(set(rigid)) != len(rigid):
            raise ValueError(
                f"segment {self.name!r}: landmarks must be distinct, got {rigid!r}"
            )
        if any(not p or not isinstance(p, str) for p in rigid):
            raise ValueError(
                f"segment {self.name!r}: landmarks entries must be non-empty "
                f"strings, got {rigid!r}"
            )

        if self.origin_landmark not in rigid:
            raise ValueError(
                f"segment {self.name!r}: origin_landmark {self.origin_landmark!r} "
                f"is not in landmarks {rigid!r}"
            )

        if self.rigid_with_parent and self.parent is None:
            raise ValueError(
                f"segment {self.name!r}: rigid_with_parent requires a parent — "
                f"a root segment cannot inherit a pose"
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
        if not any(a.kind is AxisKind.EXACT for a in axes):
            raise ValueError(
                f"segment {self.name!r}: at least one axis must be EXACT, got {axes!r}"
            )

        for a in axes:
            if not a.target_landmark or not isinstance(a.target_landmark, str):
                raise ValueError(
                    f"segment {self.name!r}: axis {a.axis!r} has a non-empty-string "
                    f"target_landmark requirement, got {a.target_landmark!r}"
                )
            if a.target_landmark == self.origin_landmark:
                raise ValueError(
                    f"segment {self.name!r}: axis {a.axis!r} target_landmark is "
                    f"the origin landmark {a.target_landmark!r}. A segment vector "
                    f"from the origin to itself would be zero-length and no "
                    f"direction could be resolved from it."
                )
            if a.target_landmark not in rigid:
                raise ValueError(
                    f"segment {self.name!r}: axis {a.axis!r} ({a.kind.value}) "
                    f"target_landmark {a.target_landmark!r} is not in landmarks "
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

    @property
    def exact_axis(self) -> AxisDefinition:
        """The segment's EXACT axis declaration (the defining direction).

        The segment's frame derives its defining direction from this axis,
        whichever local basis name (x/y/z) it is declared on. Load-time
        validation guarantees at least one EXACT axis exists; this raises
        loudly if that invariant is ever violated.
        """
        for a in self.axes:
            if a.kind is AxisKind.EXACT:
                return a
        raise ValueError(f"segment {self.name!r} has no EXACT axis")

    @property
    def approximate_axis(self) -> AxisDefinition | None:
        """The segment's APPROXIMATE axis declaration, or ``None`` if twist-less."""
        return next(
            (a for a in self.axes if a.kind is AxisKind.APPROXIMATE), None
        )

    def required_landmarks(self) -> set[str]:
        """Every landmark this segment needs to be solvable.

        The union over every field that references a landmark by name:
        ``landmarks ∪ {origin_landmark} ∪ all axis targets``. Every axis's
        ``target_landmark`` and the ``origin_landmark`` are already members of
        ``landmarks`` (enforced at load), so the set is exactly
        ``landmarks``.
        """
        names = set(self.landmarks)
        names.add(self.origin_landmark)
        for a in self.axes:
            names.add(a.target_landmark)
        return names
