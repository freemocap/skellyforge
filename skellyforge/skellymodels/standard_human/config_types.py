"""Typed shapes for the standard-human YAML config.

Each config class builds from a dict by UNPACKING it (cls(**data)) - no
string-key indexing - and coerces YAML lists into tuples in __post_init__.
A missing or misspelled key fails the load at that line.

Parts are the composability unit: a midline part is used once, a sided part is
instantiated left + right (mirroring Y) by the loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from skellyforge.skellymodels.standard_human.anatomical_landmark import LandmarkConfig
from skellyforge.skellymodels.standard_human.chain_config import ChainConfig
from skellyforge.skellymodels.standard_human.segment_definition import SegmentConfig


@dataclass(frozen=True, slots=True)
class PartConfig:
    sided: bool = False
    prefixes:list[str] = field(default_factory=list) # if list is not empty, creates multiple parts after `sided` split (e.g. for right_thumb_cmc, etc)
    landmarks: dict[str, LandmarkConfig] = field(default_factory=dict)
    segments: dict[str, SegmentConfig] = field(default_factory=dict)
    chains: dict[str, ChainConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml_path(cls, path: str|Path) -> "PartConfig":
        if not Path(path).exists():
            raise FileNotFoundError(f'{path} does not exist')
        with open(path, "r") as stream:
            data = yaml.safe_load(stream)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PartConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "landmarks",
            {
                n: c if isinstance(c, LandmarkConfig) else LandmarkConfig.from_dict(c)
                for n, c in self.landmarks.items()
            },
        )
        object.__setattr__(
            self,
            "segments",
            {
                n: c if isinstance(c, SegmentConfig) else SegmentConfig.from_dict(c)
                for n, c in self.segments.items()
            },
        )
        object.__setattr__(
            self,
            "chains",
            {
                n: c if isinstance(c, ChainConfig) else ChainConfig.from_dict(c)
                for n, c in self.chains.items()
            },
        )


@dataclass(frozen=True, slots=True)
class SkeletonConfig:
    name: str
    parts: dict[str, PartConfig]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SkeletonConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parts",
            {
                n: c if isinstance(c, PartConfig) else PartConfig.from_dict(c)
                for n, c in self.parts.items()
            },
        )


@dataclass(frozen=True, slots=True)
class FaceConfig:
    name: str
    blendshapes: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FaceConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "blendshapes",
            {k: float(v) for k, v in self.blendshapes.items()},
        )
