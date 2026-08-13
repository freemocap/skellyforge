"""Standard human model — the canonical VRM-1.0-aligned humanoid, composed.

A frozen dataclass holding the composed 55-segment human: parts authored once
(body midline, body limb, hand ×2, face), expanded into one flat indexed
segment list at load, with dict-backed name→segment and parent→children
indices built once (the per-frame O(n) scans of the old model are gone).

One model describes ONE human (SF-AL A5); multi-subject is a list of models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skellyforge.skellymodels.standard_human.body_part import (
    BODY_LIMB_PART,
    BODY_MIDLINE_PART,
)
from skellyforge.skellymodels.standard_human.face_part import FACE_PART
from skellyforge.skellymodels.standard_human.hand_part import HAND_PART
from skellyforge.skellymodels.standard_human.human_blendshapes import (
    get_blendshape_names,
)
from skellyforge.skellymodels.standard_human.segment_definition import (
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import (
    SegmentPart,
    compose_parts,
)


@dataclass(frozen=True)
class StandardHuman:
    """The canonical humanoid: one human, composed from parts."""

    name: str
    parts: tuple[tuple[SegmentPart, str], ...]
    blendshape_channels: tuple[str, ...] = field(
        default_factory=lambda: tuple(get_blendshape_names())
    )

    _segments: tuple[SegmentDefinition, ...] = field(init=False, repr=False)
    _segment_by_name: dict[str, SegmentDefinition] = field(init=False, repr=False)
    _children_by_parent: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        segments = tuple(compose_parts(self.parts))
        by_name = {s.name: s for s in segments}

        roots = [s for s in segments if s.parent is None]
        if len(roots) != 1:
            raise ValueError(
                f"standard human {self.name!r} must have exactly one root segment "
                f"(parent is None), got {len(roots)}"
            )

        for s in segments:
            if s.parent is not None and s.parent not in by_name:
                raise ValueError(
                    f"segment {s.name!r} references parent {s.parent!r}, "
                    f"which is not in the composed human"
                )

        # no cycles: every parent chain terminates at the root
        for s in segments:
            visited: set[str] = set()
            current: SegmentDefinition | None = s
            while current is not None:
                if current.name in visited:
                    raise ValueError(
                        f"cycle detected in segment hierarchy at {current.name!r}"
                    )
                visited.add(current.name)
                current = by_name[current.parent] if current.parent is not None else None

        children: dict[str, tuple[str, ...]] = {name: () for name in by_name}
        for s in segments:
            if s.parent is not None:
                children[s.parent] = (*children[s.parent], s.name)

        object.__setattr__(self, "_segments", segments)
        object.__setattr__(self, "_segment_by_name", by_name)
        object.__setattr__(self, "_children_by_parent", children)

    @property
    def segments(self) -> tuple[SegmentDefinition, ...]:
        """All segments in hierarchy order (authoring order)."""
        return self._segments

    @property
    def segment_names(self) -> list[str]:
        return [s.name for s in self._segments]

    @property
    def segment_parents(self) -> dict[str, str | None]:
        return {s.name: s.parent for s in self._segments}

    @property
    def joint_hierarchy(self) -> dict[str, list[str]]:
        """Parent → children over segments."""
        return {p: list(c) for p, c in self._children_by_parent.items()}

    @property
    def root_segment(self) -> SegmentDefinition:
        return self._segments[0]  # authoring order puts the root first

    def get_children(self, segment_name: str) -> list[SegmentDefinition]:
        return [
            self._segment_by_name[n]
            for n in self._children_by_parent.get(segment_name, ())
        ]

    def get_segment_chain(self, segment_name: str) -> list[SegmentDefinition]:
        """The chain from the root to the named segment (inclusive)."""
        chain: list[SegmentDefinition] = []
        current = self._segment_by_name[segment_name]
        while True:
            chain.append(current)
            if current.parent is None:
                break
            current = self._segment_by_name[current.parent]
        chain.reverse()
        return chain

    def required_keypoints(self) -> set[str]:
        """Every keypoint the segments need (Task 6's contract set).

        Every segment is driven and enters its keypoints here; the union over
        all 55 segments is what a tracker must be able to supply.
        """
        required: set[str] = set()
        for s in self._segments:
            required |= s.required_keypoints()
        return required


def compose_standard_human(name: str = "standard_human") -> StandardHuman:
    """The standard 55-segment human: body + both hands + the face."""
    return StandardHuman(
        name=name,
        parts=(
            (BODY_MIDLINE_PART, ""),
            (BODY_LIMB_PART, "left_"),
            (BODY_LIMB_PART, "right_"),
            (HAND_PART, "left_"),
            (HAND_PART, "right_"),
            (FACE_PART, ""),
        ),
    )
