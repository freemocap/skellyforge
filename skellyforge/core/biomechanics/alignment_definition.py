"""Model-authored anatomical evidence for scene-reference estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentDefinition:
    """Resolved model objects; neither detector names nor filename conventions."""

    body_regions: tuple[RigidBodySegment, ...]
    foot_contacts: tuple[AnatomicalLandmark, ...]

    def __post_init__(self) -> None:
        if not self.body_regions:
            raise ValueError("Alignment definition requires at least one body region")
        if len({region.name for region in self.body_regions}) != len(self.body_regions):
            raise ValueError("Alignment body regions must be unique")
        if len({contact.name for contact in self.foot_contacts}) != len(
            self.foot_contacts
        ):
            raise ValueError("Alignment foot contacts must be unique")

    @classmethod
    def from_yaml(
        cls, *, path: Path, skeleton: SkeletonDefinition
    ) -> AlignmentDefinition:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "body_regions",
            "foot_contacts",
        }:
            raise ValueError(
                "Alignment YAML requires exactly body_regions and foot_contacts"
            )
        for names in document.values():
            if not isinstance(names, list) or any(
                not isinstance(name, str) for name in names
            ):
                raise ValueError(
                    "Alignment entries must be lists of canonical model names"
                )
        return cls(
            body_regions=tuple(
                skeleton.segments[name] for name in document["body_regions"]
            ),
            foot_contacts=tuple(
                skeleton.landmarks[name] for name in document["foot_contacts"]
            ),
        )

    @classmethod
    def from_default_human(cls, *, skeleton: SkeletonDefinition) -> AlignmentDefinition:
        return cls.from_yaml(
            path=Path(__file__).resolve().parents[2]
            / "definitions"
            / "human_skeleton"
            / "alignment_definition.yaml",
            skeleton=skeleton,
        )
