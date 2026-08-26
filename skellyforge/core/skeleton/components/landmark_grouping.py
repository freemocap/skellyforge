"""Named sets of landmarks, and named sets of edges between them.

A skeleton's segments say how it MOVES. These say what it LOOKS LIKE: which landmarks
belong together (a face, a board's charuco corners) and which pairs of them a consumer
should draw as edges (a skull outline, a board's grid, the four sides of a marker).

They exist because the alternative is a consumer recovering that structure by parsing
names - splitting `ArucoMarkerCorner-{id}-{corner}` back into a quad, or colouring a point
because its name starts with `aruco`. That works until a name changes, and it silently
puts the model's structure in two places: the skeleton, and a regex somewhere downstream.
Structure travels in the model. Names are opaque identifiers.

Groups carry TAGS rather than colours. A tag says what a group is ("left", "hand",
"aruco_marker"); a palette says what that should look like. Keeping the two apart is what
lets a user recolour their whole skeleton by editing one mapping instead of re-authoring
every component, and it keeps a presentation choice out of a geometry file.

Tags are ordered, and a palette resolves the FIRST one it knows - so a group tagged
`["left_hand", "left"]` takes the hand colour where a palette defines one and falls back to
the left-side colour where it does not. That is how "left/hand is cyan but left is blue"
is expressed without the model needing to know either colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from skellyforge.type_overloads import LandmarkNameString

LANDMARKS_PER_CONNECTION: Final[int] = 2


def _validated_tags(*, owner: str, tags: object) -> tuple[str, ...]:
    """An ordered tuple of non-empty tag strings, or empty. Anything else is refused."""
    if tags is None:
        return ()
    if isinstance(tags, str) or not isinstance(tags, Sequence):
        raise ValueError(
            f"{owner}: `tags` must be a list of tag names - got {tags!r}. A single tag is "
            "still a list of one, so that the order a palette resolves them in is visible."
        )
    parsed = tuple(str(tag) for tag in tags)
    if any(not tag for tag in parsed):
        raise ValueError(f"{owner}: tags must be non-empty strings - got {parsed}")
    duplicates = sorted({tag for tag in parsed if parsed.count(tag) > 1})
    if duplicates:
        raise ValueError(f"{owner}: lists {duplicates} more than once")
    return parsed


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkGroup:
    """A named set of landmarks that belong together.

    Attributes:
        name: what this group is called, unique within its skeleton.
        landmark_names: the landmarks in it. Membership is validated against the skeleton,
            not here - a group is loaded from one component and may name a landmark that
            another component declares.
        tags: what this group IS, most specific first. A palette resolves the first tag
            it knows into a colour; a group with no tags, or none the palette knows, gets
            the palette's default.
    """

    name: str
    landmark_names: tuple[LandmarkNameString, ...]
    tags: tuple[str, ...] = ()

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
            self, "tags", _validated_tags(owner=f"landmark group {self.name!r}", tags=self.tags)
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
        tags: what this group IS, most specific first — resolved to a colour by a palette.
    """

    name: str
    pairs: tuple[tuple[LandmarkNameString, ...], ...]
    tags: tuple[str, ...] = ()

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
            self, "tags", _validated_tags(owner=f"connection group {self.name!r}", tags=self.tags)
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
    unexpected = sorted(set(entry) - {"landmark_names", "tags"})
    if unexpected:
        raise ValueError(
            f"landmark group {name!r}: unexpected keys {unexpected} - expected "
            "['landmark_names', 'tags']"
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
        tags=_validated_tags(owner=f"landmark group {name!r}", tags=entry.get("tags")),
    )


def build_landmark_connection_group(*, name: str, entry: object) -> LandmarkConnectionGroup:
    """One authored `landmark_connections:` entry as a `LandmarkConnectionGroup`."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"connection group {name!r} must be a mapping with `pairs:` - got "
            f"{type(entry).__name__}"
        )
    unexpected = sorted(set(entry) - {"pairs", "tags"})
    if unexpected:
        raise ValueError(
            f"connection group {name!r}: unexpected keys {unexpected} - expected "
            "['pairs', 'tags']"
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
        name=name,
        pairs=tuple(parsed),
        tags=_validated_tags(owner=f"connection group {name!r}", tags=entry.get("tags")),
    )
