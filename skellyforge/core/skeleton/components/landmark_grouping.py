"""Named sets of landmarks, and named sets of edges between them.

A skeleton's segments say how it MOVES. These say what it LOOKS LIKE: which landmarks
belong together (a face, a board's charuco corners) and which pairs of them a consumer
should draw as edges (a skull outline, a board's grid, the four sides of a marker).

They exist because the alternative is a consumer recovering that structure by parsing
names - splitting `ArucoMarkerCorner-{id}-{corner}` back into a quad, or colouring a point
because its name starts with `aruco`. That works until a name changes, and it silently
puts the model's structure in two places: the skeleton, and a regex somewhere downstream.
Structure travels in the model. Names are opaque identifiers.

Colour is authored here rather than chosen by the consumer for the same reason. A charuco
grid is green and its markers are orange because the model says so, which is what lets one
renderer draw a skull and a calibration board without knowing which is which. A consumer
that wants its own palette is free to ignore it; a consumer that does not want to invent
one has an answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from skellyforge.type_overloads import LandmarkNameString

LANDMARKS_PER_CONNECTION: Final[int] = 2
HEX_COLOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"#[0-9a-fA-F]{6}")


def _validated_color(*, owner: str, color: object) -> str | None:
    """A `#rrggbb` string, or None. Anything else is refused rather than passed through.

    A colour that reaches a renderer malformed is a silent no-op there, so it is checked
    at load where the offending group can be named.
    """
    if color is None:
        return None
    if not isinstance(color, str) or not HEX_COLOR_PATTERN.fullmatch(color):
        raise ValueError(
            f"{owner}: color must be a '#rrggbb' hex string or absent - got {color!r}"
        )
    return color


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkGroup:
    """A named set of landmarks that belong together.

    Attributes:
        name: what this group is called, unique within its skeleton.
        landmark_names: the landmarks in it. Membership is validated against the skeleton,
            not here - a group is loaded from one component and may name a landmark that
            another component declares.
        color: an optional authored `#rrggbb` for consumers that draw these points.
    """

    name: str
    landmark_names: tuple[LandmarkNameString, ...]
    color: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("landmark group name must be non-empty")
        if not self.landmark_names:
            raise ValueError(
                f"landmark group {self.name!r} is empty - a group that names nothing is a "
                "leftover, not a grouping"
            )
        duplicates = sorted(
            {name for name in self.landmark_names if self.landmark_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"landmark group {self.name!r} lists {duplicates} more than once"
            )
        object.__setattr__(
            self, "color", _validated_color(owner=f"landmark group {self.name!r}", color=self.color)
        )


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkConnectionGroup:
    """A named set of landmark pairs a consumer should draw as edges.

    Distinct from a skeleton's `connections`, which join SEGMENT origins along the joint
    tree. These join landmarks, which is the only kind of edge a one-segment skeleton has.

    Attributes:
        name: what this group is called, unique within its skeleton.
        pairs: the `(landmark, landmark)` edges. Annotated as variadic inner tuples rather
            than as 2-tuples so that a wrong arity is refused by `__post_init__` with a
            message naming this group, instead of by a type-check error that names only a
            hint. Membership is validated against the skeleton, not here.
        color: an optional authored `#rrggbb` for consumers that draw these edges.
    """

    name: str
    pairs: tuple[tuple[LandmarkNameString, ...], ...]
    color: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("landmark connection group name must be non-empty")
        if not self.pairs:
            raise ValueError(
                f"connection group {self.name!r} has no pairs - a group that connects "
                "nothing is a leftover, not a grouping"
            )
        malformed = [pair for pair in self.pairs if len(pair) != LANDMARKS_PER_CONNECTION]
        if malformed:
            raise ValueError(
                f"connection group {self.name!r}: every entry must be exactly "
                f"{LANDMARKS_PER_CONNECTION} landmark names - got {malformed}"
            )
        self_edges = sorted({pair[0] for pair in self.pairs if pair[0] == pair[1]})
        if self_edges:
            raise ValueError(
                f"connection group {self.name!r}: these landmarks are connected to "
                f"themselves - {self_edges}"
            )
        object.__setattr__(
            self, "color", _validated_color(owner=f"connection group {self.name!r}", color=self.color)
        )

    @property
    def landmark_names(self) -> frozenset[LandmarkNameString]:
        """Every landmark this group's edges touch."""
        return frozenset(name for pair in self.pairs for name in pair)


def build_landmark_group(*, name: str, entry: object) -> LandmarkGroup:
    """One authored `landmark_groups:` entry as a `LandmarkGroup`."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"landmark group {name!r} must be a mapping with `landmark_names:` - got "
            f"{type(entry).__name__}"
        )
    unexpected = sorted(set(entry) - {"landmark_names", "color"})
    if unexpected:
        raise ValueError(
            f"landmark group {name!r}: unexpected keys {unexpected} - expected "
            "['color', 'landmark_names']"
        )
    landmark_names = entry.get("landmark_names")
    if not isinstance(landmark_names, Sequence) or isinstance(landmark_names, (str, bytes)):
        raise ValueError(
            f"landmark group {name!r}: `landmark_names` must be a list - got "
            f"{landmark_names!r}"
        )
    return LandmarkGroup(
        name=name,
        landmark_names=tuple(str(member) for member in landmark_names),
        color=entry.get("color"),
    )


def build_landmark_connection_group(*, name: str, entry: object) -> LandmarkConnectionGroup:
    """One authored `landmark_connections:` entry as a `LandmarkConnectionGroup`."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"connection group {name!r} must be a mapping with `pairs:` - got "
            f"{type(entry).__name__}"
        )
    unexpected = sorted(set(entry) - {"pairs", "color"})
    if unexpected:
        raise ValueError(
            f"connection group {name!r}: unexpected keys {unexpected} - expected "
            "['color', 'pairs']"
        )
    pairs = entry.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ValueError(
            f"connection group {name!r}: `pairs` must be a list of landmark pairs - got "
            f"{pairs!r}"
        )
    parsed: list[tuple[LandmarkNameString, ...]] = []
    for pair in pairs:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
            raise ValueError(
                f"connection group {name!r}: every entry must be a pair of landmark "
                f"names - got {pair!r}"
            )
        parsed.append(tuple(str(member) for member in pair))
    return LandmarkConnectionGroup(
        name=name, pairs=tuple(parsed), color=entry.get("color")
    )
