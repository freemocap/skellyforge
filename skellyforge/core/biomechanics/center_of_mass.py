"""Center of mass per anatomical segment, authored as weighted sums of landmarks.

The de Leva table gives each segment's COM as a 1D fraction of segment length, which puts
a limb's COM on the bone (correct - the bone is the mass) but puts the trunk's COM on the
spine (wrong - the trunk's mass is soft tissue anterior to it). This module is the fix: it
reads, per anatomical segment, a weighted sum of the skeleton's own landmarks, so the
abdomen COM is placed by anterior landmarks rather than by the lumbar spine.

It also resolves each segment's proximal/distal anchors, which segment_inertia.py uses to
orient and scale the inertia tensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from skellyforge.core.biomechanics import FRACTION_SUM_TOLERANCE
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose
from skellyforge.type_overloads import FloatArray


@dataclass(frozen=True, slots=True)
class ComLandmarkWeight:
    """One landmark and the weight it carries in a COM sum.

    Attributes:
        landmark: the landmark name, unsided (prefixed left_/right_ for sided segments).
        weight: the weight this landmark carries; all weights on a segment sum to 1.0.
    """

    landmark: str
    weight: float


@dataclass(frozen=True, slots=True, eq=False)
class SegmentComDefinition:
    """One anatomical segment's COM and long-axis definition, in skeleton-landmark terms.

    Attributes:
        name: snake_case anatomical segment name (e.g. "upper_arm", "middle_trunk").
        proximal: the unsided landmark at the segment's proximal (long-axis) end.
        distal: the unsided landmark at the segment's distal end.
        weights: the landmarks and weights whose weighted sum is the segment's COM.
        sided: whether the segment exists as a left_/right_ pair.
    """

    name: str
    proximal: str
    distal: str
    weights: tuple[ComLandmarkWeight, ...]
    sided: bool = False

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(f"segment {self.name!r}: needs at least one COM landmark")
        total = sum(entry.weight for entry in self.weights)
        if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
            raise ValueError(
                f"segment {self.name!r}: COM weights must sum to 1.0 - got {total}"
            )
        for entry in self.weights:
            if entry.weight <= 0.0:
                raise ValueError(
                    f"segment {self.name!r}: COM weights must be positive - got "
                    f"{entry.weight} for {entry.landmark!r}"
                )

    @property
    def side_entries(self) -> tuple[tuple[str, str | None], ...]:
        """The (full segment name, side) pairs this definition resolves to.

        A sided definition resolves to two entries (left_/right_); a midline one to one.
        """
        if self.sided:
            return ((f"left_{self.name}", "left"), (f"right_{self.name}", "right"))
        return ((self.name, None),)

    def resolve_landmark(self, *, side: str | None, landmark: str) -> str:
        """The skeleton landmark name for an unsided landmark, given the side."""
        if self.sided:
            if side not in ("left", "right"):
                raise ValueError(
                    f"segment {self.name!r} is sided - side must be 'left' or 'right', "
                    f"got {side!r}"
                )
            return f"{side}_{landmark}"
        return landmark


@dataclass(frozen=True, slots=True, eq=False)
class CenterOfMassDefinitions:
    """Every anatomical segment's COM definition, keyed by anatomical name."""

    definitions: Mapping[str, SegmentComDefinition]

    def __post_init__(self) -> None:
        if not self.definitions:
            raise ValueError("center-of-mass definitions need at least one segment")
        for name, definition in self.definitions.items():
            if name != definition.name:
                raise ValueError(
                    f"COM definitions must be keyed by their own name - got key {name!r} "
                    f"for segment {definition.name!r}"
                )

    def get(self, *, name: str) -> SegmentComDefinition:
        """The COM definition named by name, failing loudly on an unknown name."""
        try:
            return self.definitions[name]
        except KeyError as error:
            raise KeyError(
                f"unknown COM definition {name!r} - known definitions are "
                f"{sorted(self.definitions)}"
            ) from error

    @property
    def all_segment_names(self) -> tuple[str, ...]:
        """Every full (sided) segment name these definitions resolve to, sorted."""
        return tuple(
            sorted(
                full_name
                for definition in self.definitions.values()
                for full_name, _ in definition.side_entries
            )
        )

    def validate_against(self, *, skeleton: SkeletonDefinition) -> None:
        """Fail loudly if any COM or anchor landmark does not exist in the skeleton."""
        known = set(skeleton.landmarks)
        for definition in self.definitions.values():
            for _, side in definition.side_entries:
                referenced = [definition.proximal, definition.distal] + [
                    entry.landmark for entry in definition.weights
                ]
                for landmark in referenced:
                    resolved = definition.resolve_landmark(side=side, landmark=landmark)
                    if resolved not in known:
                        raise ValueError(
                            f"COM definition for {definition.name!r} references landmark "
                            f"{resolved!r}, which the skeleton {skeleton.name!r} does "
                            f"not have"
                        )

    @classmethod
    def from_yaml(cls, *, path: Path) -> CenterOfMassDefinitions:
        """Load COM definitions from a YAML file in the shipped shape."""
        if not path.is_file():
            raise FileNotFoundError(f"center-of-mass YAML {path} is not a file")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"{path} must parse to a mapping, got {type(document).__name__}")
        return cls.from_document(document=document, source=str(path))

    @classmethod
    def from_default_yaml(cls) -> CenterOfMassDefinitions:
        """Load the shipped COM definitions for the standard human."""
        path = (
            Path(__file__).resolve().parents[2]
            / "definitions"
            / "human_skeleton"
            / "center_of_mass.yaml"
        )
        return cls.from_yaml(path=path)

    @classmethod
    def from_document(
        cls, *, document: Mapping[str, object], source: str
    ) -> CenterOfMassDefinitions:
        """Build definitions from an already-parsed mapping (the shape from_yaml reads)."""
        entries = document.get("segments")
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError(f"{source}: needs a non-empty 'segments' mapping")

        definitions = {
            str(name): _build_definition(name=str(name), entry=entry)
            for name, entry in entries.items()
        }
        return cls(definitions=definitions)


_DEFINITION_KEYS: frozenset[str] = frozenset(
    {"proximal", "distal", "landmarks", "sided"}
)


def _build_definition(*, name: str, entry: object) -> SegmentComDefinition:
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"COM segment {name!r} must be a mapping, got {type(entry).__name__}"
        )
    unexpected_keys = set(entry) - _DEFINITION_KEYS
    if unexpected_keys:
        raise ValueError(
            f"COM segment {name!r}: unexpected keys {sorted(unexpected_keys)} - expected "
            f"{sorted(_DEFINITION_KEYS)}"
        )

    proximal = entry.get("proximal")
    distal = entry.get("distal")
    for label, value in (("proximal", proximal), ("distal", distal)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"COM segment {name!r}: '{label}' must be a non-empty string")

    sided = entry.get("sided", False)
    if not isinstance(sided, bool):
        raise ValueError(
            f"COM segment {name!r}: 'sided' must be a boolean, got {type(sided).__name__}"
        )

    weights = _build_weights(name=name, node=entry.get("landmarks"))
    return SegmentComDefinition(
        name=name,
        proximal=str(proximal),
        distal=str(distal),
        weights=weights,
        sided=sided,
    )


def _build_weights(*, name: str, node: object) -> tuple[ComLandmarkWeight, ...]:
    if node is None:
        raise ValueError(f"COM segment {name!r}: needs a 'landmarks' list")
    if isinstance(node, str):
        raise ValueError(
            f"COM segment {name!r}: 'landmarks' must be a list of names or of "
            f"{'landmark': ..., 'weight': ...} mappings, got a bare string"
        )
    if not isinstance(node, Sequence) or not node:
        raise ValueError(f"COM segment {name!r}: 'landmarks' must be a non-empty list")

    entries: list[ComLandmarkWeight] = []
    for item in node:
        if isinstance(item, str):
            entries.append(ComLandmarkWeight(landmark=item, weight=0.0))
        elif isinstance(item, Mapping):
            landmark = item.get("landmark")
            weight = item.get("weight")
            if not isinstance(landmark, str) or not landmark:
                raise ValueError(
                    f"COM segment {name!r}: each weighted landmark needs a non-empty "
                    f"'landmark' name"
                )
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"COM segment {name!r}: landmark {landmark!r} needs a numeric 'weight'"
                )
            entries.append(ComLandmarkWeight(landmark=landmark, weight=float(weight)))
        else:
            raise ValueError(
                f"COM segment {name!r}: each landmark must be a name or a "
                f"{'landmark': ..., 'weight': ...} mapping, got {type(item).__name__}"
            )

    # Bare names (no explicit weight) mean "mean of the defining landmarks": equal
    # weights. Explicit weights are kept as-is and validated to sum to 1 by the dataclass.
    unweighted = [entry for entry in entries if entry.weight == 0.0]
    if unweighted:
        if len(unweighted) != len(entries):
            raise ValueError(
                f"COM segment {name!r}: landmarks are either all bare names (mean) or all "
                f"weighted - do not mix the two"
            )
        equal = 1.0 / len(entries)
        entries = [
            ComLandmarkWeight(landmark=entry.landmark, weight=equal) for entry in entries
        ]
    return tuple(entries)


def landmark_world_positions(
    *, skeleton: SkeletonDefinition, pose: SkeletonPose
) -> dict[str, FloatArray]:
    """Every landmark's world position at this pose, keyed by canonical name.

    A landmark's world position is its local position rotated by its segment's
    orientation and translated to that segment's world origin. Segments absent
    from a partial pose (occluded) are omitted, so their landmarks are absent
    too — downstream consumers roll up over the visible ones.
    """
    positions: dict[str, FloatArray] = {}
    for segment_name, segment in skeleton.segments.items():
        segment_pose = pose.segment_poses.get(segment_name)
        if segment_pose is None:
            continue
        origin = segment_pose.origin.array
        for landmark_name, landmark in segment.landmarks.items():
            positions[landmark_name] = origin + segment_pose.orientation.rotate_vector(
                vector=landmark.local_position.array
            )
    return positions


def segment_com(
    *, definition: SegmentComDefinition, side: str | None, world: Mapping[str, FloatArray]
) -> FloatArray | None:
    """One segment's world COM as the weighted sum of its defining landmarks.

    Occluded landmarks (absent from ``world``) are skipped and the remaining
    weights re-normalized, so the COM is the weighted mean of the VISIBLE
    landmarks. Returns ``None`` when every defining landmark is occluded.
    """
    total = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    for entry in definition.weights:
        resolved = definition.resolve_landmark(side=side, landmark=entry.landmark)
        position = world.get(resolved)
        if position is None:
            continue
        total = total + entry.weight * position
        total_weight += entry.weight
    if total_weight <= 0.0:
        return None
    return total / total_weight


def compute_segment_coms(
    *, definitions: CenterOfMassDefinitions, world: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    """Every segment's world COM, keyed by its full (sided) segment name.

    Segments whose defining landmarks are all occluded are omitted.
    """
    result: dict[str, FloatArray] = {}
    for definition in definitions.definitions.values():
        for full_name, side in definition.side_entries:
            com = segment_com(definition=definition, side=side, world=world)
            if com is not None:
                result[full_name] = com
    return result


def segment_anchors(
    *, definition: SegmentComDefinition, side: str | None, world: Mapping[str, FloatArray]
) -> tuple[FloatArray, FloatArray]:
    """The proximal and distal world positions that define a segment's long axis."""
    proximal = definition.resolve_landmark(side=side, landmark=definition.proximal)
    distal = definition.resolve_landmark(side=side, landmark=definition.distal)
    try:
        return world[proximal], world[distal]
    except KeyError as error:
        raise KeyError(
            f"COM definition for {definition.name!r} references an anchor landmark with "
            f"no world position"
        ) from error
