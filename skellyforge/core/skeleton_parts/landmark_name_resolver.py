"""Folds every name a landmark answers to down to its one canonical name.

This is the boundary where alias handling lives, and the ONLY place it lives. Build the
resolver once at setup, use it to normalize incoming names once - when constructing a
`PointRingBuffer`, or when adapting a mapping of observed positions - and everything
downstream sees canonical names only. Nothing in `geometry/` knows the word "alias", and
nothing resolves a name inside a per-frame loop.

Building the resolver is also where GLOBAL alias uniqueness is enforced, because a single
landmark cannot check a property of the whole set from inside itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.type_overloads import LandmarkNameString


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkNameResolver:
    """Maps any known landmark name - canonical or alias - to its canonical name.

    Attributes:
        canonical_name_by_known_name: every name the landmark set answers to, mapped to
            the canonical name it belongs to. Canonical names map to themselves.
    """

    canonical_name_by_known_name: Mapping[LandmarkNameString, LandmarkNameString]

    @classmethod
    def from_landmarks(
        cls, *, landmarks: Iterable[AnatomicalLandmark]
    ) -> LandmarkNameResolver:
        """Index a landmark set, failing loudly on any name or alias claimed twice.

        Raises:
            ValueError: two landmarks answer to the same name, whether as canonical names,
                as aliases, or as one of each.
        """
        canonical_name_by_known_name: dict[LandmarkNameString, LandmarkNameString] = {}
        for landmark in landmarks:
            for known_name in landmark.all_names:
                claimed_by = canonical_name_by_known_name.get(known_name)
                if claimed_by is not None:
                    raise ValueError(
                        f"Landmark names and aliases must be globally unique - "
                        f"{known_name!r} is claimed by both {claimed_by!r} and "
                        f"{landmark.name!r}"
                    )
                canonical_name_by_known_name[known_name] = landmark.name
        if not canonical_name_by_known_name:
            raise ValueError("Cannot build a LandmarkNameResolver from no landmarks")
        return cls(canonical_name_by_known_name=canonical_name_by_known_name)

    def resolve(self, *, name: LandmarkNameString) -> LandmarkNameString:
        """The canonical name for `name`, which may itself be canonical or an alias."""
        canonical_name = self.canonical_name_by_known_name.get(name)
        if canonical_name is None:
            raise KeyError(
                f"Unknown landmark name {name!r} - known names are "
                f"{sorted(self.canonical_name_by_known_name)}"
            )
        return canonical_name

    def resolve_all(self, *, names: Sequence[LandmarkNameString]) -> tuple[LandmarkNameString, ...]:
        """Canonical names for a sequence of names, in order.

        Use this once when choosing a `PointRingBuffer`'s `point_names`, so the buffer is
        keyed canonically and no resolution happens per frame.
        """
        return tuple(self.resolve(name=name) for name in names)

    def resolve_points(
        self, *, points: Mapping[LandmarkNameString, Point]
    ) -> dict[LandmarkNameString, Point]:
        """Re-key a mapping of observed positions by canonical name.

        Raises:
            KeyError: a key is not a known name or alias.
            ValueError: two keys resolve to the same canonical name.
        """
        resolved: dict[LandmarkNameString, Point] = {}
        source_name_by_canonical_name: dict[LandmarkNameString, LandmarkNameString] = {}
        for name, point in points.items():
            canonical_name = self.resolve(name=name)
            already_supplied_by = source_name_by_canonical_name.get(canonical_name)
            if already_supplied_by is not None:
                raise ValueError(
                    f"Two keys resolve to the landmark {canonical_name!r} - "
                    f"{already_supplied_by!r} and {name!r}"
                )
            source_name_by_canonical_name[canonical_name] = name
            resolved[canonical_name] = point
        return resolved
