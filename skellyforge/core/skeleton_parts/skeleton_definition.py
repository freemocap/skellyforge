from dataclasses import dataclass, field
from pathlib import Path

import yaml
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment

SkeletonNameString = str


@dataclass(frozen=True, slots=True)
class SkeletonDefinition:
    name: SkeletonNameString
    landmarks: list[AnatomicalLandmark] = field(default_factory=list)
    segments: list[RigidBodySegment] = field(default_factory=list)
    # linkages: list[SegmentLinkage] = field(default_factory=list)
    # chains: list[KinematicChain] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path:str|Path) -> 'SkeletonDefinition':
        if not Path(yaml_path).is_file():
            raise FileNotFoundError(f"{yaml_path} is not a file")
        if not Path(yaml_path).suffix == ".yaml":
            raise FileNotFoundError(f"{yaml_path} is not a .yaml file")
        with open(yaml_path, "r") as stream:
            data = yaml.safe_load(stream)

        return SkeletonDefinition(**data)
