"""The skeleton: a collection of chains composing one standard human.

Defined in YAML and compiled by from_yaml. A part is the composability unit: a
midline part is used once; a sided part is instantiated left + right (the right
side mirrors Y). References are resolved to OBJECTS after load - never strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import yaml

from skellyforge.skellymodels.standard_human.anatomical_landmark import (
    AnatomicalLandmark,
)
from skellyforge.skellymodels.standard_human.config_types import (
    AxisConfig,
    ChainConfig,
    LandmarkConfig,
    PartConfig,
    SegmentConfig,
    SkeletonConfig,
)
from skellyforge.skellymodels.standard_human.joint_linkage import JointLinkage
from skellyforge.skellymodels.standard_human.kinematic_chain import KinematicChain
from skellyforge.skellymodels.standard_human.rigid_body_segment import (
    AxisDefinition,
    RigidBodySegment,
)


_Node = TypeVar("_Node")


def load_config(node: _Node, base: Path) -> _Node:
    """A dict with the single key '$include' loads that file in place; a bare
    string is always a value (never a path). Typed as a generic transform."""
    if isinstance(node, dict) and set(node) == {"$include"}:
        path = base / node["$include"]
        return cast(_Node, load_config(yaml.safe_load(path.read_text()), path.parent))
    if isinstance(node, list):
        return cast(_Node, [load_config(v, base) for v in node])
    if isinstance(node, dict):
        return cast(_Node, {k: load_config(v, base) for k, v in node.items()})
    return node


def _mirror_y(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """The standard mirror across the sagittal (XZ) plane: negate Y."""
    return (vec[0], -vec[1], vec[2])


def _prefixed(name: str, prefix: str, sided_segments: set[str]) -> str:
    """Prefix a reference when it names a segment in ANY sided part, so a foot's
    parent 'lower_leg' resolves to left_lower_leg / right_lower_leg."""
    return prefix + name if name in sided_segments else name


def _instantiate_landmark(
    config: LandmarkConfig, prefix: str, mirror: bool, sided_segments: set[str]
) -> LandmarkConfig:
    return LandmarkConfig(
        definition=config.definition,
        reference_frame=_prefixed(config.reference_frame, prefix, sided_segments),
        rest_position=_mirror_y(config.rest_position) if mirror else config.rest_position,
    )


def _instantiate_axis(
    config: AxisConfig, prefix: str, mirror: bool
) -> AxisConfig:
    return AxisConfig(
        axis=config.axis,
        target_landmark=prefix + config.target_landmark,
        rest_direction=(
            _mirror_y(config.rest_direction)
            if (mirror and config.rest_direction is not None)
            else config.rest_direction
        ),
    )


def _instantiate_segment(
    config: SegmentConfig, prefix: str, mirror: bool, sided_segments: set[str]
) -> SegmentConfig:
    return SegmentConfig(
        parent=(
            _prefixed(config.parent, prefix, sided_segments)
            if config.parent is not None
            else None
        ),
        origin_landmark=prefix + config.origin_landmark,
        landmarks=tuple(prefix + n for n in config.landmarks),
        axes=tuple(_instantiate_axis(a, prefix, mirror) for a in config.axes),
        rigid_with_parent=config.rigid_with_parent,
    )


def _instantiate_chain(
    config: ChainConfig, prefix: str, sided_segments: set[str]
) -> ChainConfig:
    return ChainConfig(
        start=_prefixed(config.start, prefix, sided_segments),
        end=_prefixed(config.end, prefix, sided_segments),
    )


def _compose_parts(
    parts: dict[str, PartConfig],
) -> tuple[
    dict[str, LandmarkConfig],
    dict[str, SegmentConfig],
    dict[str, ChainConfig],
]:
    """Expand parts into flat, prefixed (left_/right_) configs. A sided part is
    instantiated twice; the right side mirrors Y."""
    landmarks: dict[str, LandmarkConfig] = {}
    segments: dict[str, SegmentConfig] = {}
    chains: dict[str, ChainConfig] = {}

    # every sided part's segment names — the prefixing authority for references
    sided_segments: set[str] = set()
    for part in parts.values():
        if part.sided:
            sided_segments.update(part.segments)

    for part in parts.values():
        if not part.sided:
            landmarks.update(part.landmarks)
            segments.update(part.segments)
            chains.update(part.chains)
            continue

        for prefix, mirror in (("left_", False), ("right_", True)):
            for name, lc in part.landmarks.items():
                new_name = prefix + name
                if new_name not in landmarks:
                    landmarks[new_name] = _instantiate_landmark(
                        lc, prefix, mirror, sided_segments
                    )
            for name, sc in part.segments.items():
                segments[prefix + name] = _instantiate_segment(
                    sc, prefix, mirror, sided_segments
                )
            for name, cc in part.chains.items():
                chains[prefix + name] = _instantiate_chain(cc, prefix, sided_segments)

    return landmarks, segments, chains


def derive_linkages(
    segments: tuple[RigidBodySegment, ...],
) -> tuple[JointLinkage, ...]:
    """Each child's parent edge IS a linkage; the shared point is the child's origin."""
    linkages: list[JointLinkage] = []
    for child in segments:
        if child.parent is None:
            continue
        shared = child.origin_landmark
        linkages.append(
            JointLinkage(
                name=shared.name,
                parent_segment=child.parent,
                child_segment=child,
                shared_landmark=shared,
            )
        )
    return tuple(linkages)


@dataclass(frozen=True, slots=True)
class HumanSkeleton:
    name: str
    segments: tuple[RigidBodySegment, ...]
    linkages: tuple[JointLinkage, ...]
    chains: tuple[KinematicChain, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HumanSkeleton":
        path = Path(path)
        config = SkeletonConfig.from_dict(
            load_config(yaml.safe_load(path.read_text()), path.parent)
        )
        landmark_cfgs, segment_cfgs, chain_cfgs = _compose_parts(config.parts)

        landmarks: dict[str, AnatomicalLandmark] = {
            name: AnatomicalLandmark.from_config(name, c)
            for name, c in landmark_cfgs.items()
        }

        segments: dict[str, RigidBodySegment] = {}

        def build_segment(name: str) -> RigidBodySegment:
            if name in segments:
                return segments[name]
            sc = segment_cfgs[name]
            parent = build_segment(sc.parent) if sc.parent else None
            seg = RigidBodySegment(
                name=name,
                parent=parent,
                landmarks=tuple(landmarks[n] for n in sc.landmarks),
                origin_landmark=landmarks[sc.origin_landmark],
                axes=tuple(AxisDefinition.from_config(a) for a in sc.axes),
                rigid_with_parent=sc.rigid_with_parent,
            )
            segments[name] = seg
            return seg

        for name in segment_cfgs:
            build_segment(name)

        linkages = derive_linkages(tuple(segments.values()))
        chains = tuple(
            KinematicChain(
                name=n,
                start_segment=segments[c.start],
                end_segment=segments[c.end],
            )
            for n, c in chain_cfgs.items()
        )
        return cls(
            name=config.name,
            segments=tuple(segments.values()),
            linkages=linkages,
            chains=chains,
        )

    def segment(self, name: str) -> RigidBodySegment:
        return next(s for s in self.segments if s.name == name)

    def chain(self, name: str) -> KinematicChain:
        return next(c for c in self.chains if c.name == name)
