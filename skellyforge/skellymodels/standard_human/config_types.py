"""Typed shapes for the standard-human YAML config.

Each config class builds from a dict by UNPACKING it (cls(**data)) - no
string-key indexing - and coerces YAML lists into tuples in __post_init__.
A missing or misspelled key fails the load at that line.

Parts are the composability unit: a midline part is used once, a sided part is
instantiated left + right (mirroring Y) by the loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class LandmarkConfig:
    definition: str
    reference_frame: str
    rest_position: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LandmarkConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rest_position", tuple(float(v) for v in self.rest_position)
        )


@dataclass(frozen=True, slots=True)
class AxisConfig:
    axis: Literal["x", "y", "z", "-x", "-y", "-z"]
    target_landmark: str
    rest_direction: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AxisConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        if self.rest_direction is not None:
            object.__setattr__(
                self, "rest_direction", tuple(float(v) for v in self.rest_direction)
            )


@dataclass(frozen=True, slots=True)
class SegmentConfig:
    parent: str | None = None
    origin_landmark: str = ""
    landmarks: tuple[str, ...] = ()
    axes: tuple[AxisConfig, ...] = ()
    rigid_with_parent: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SegmentConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        object.__setattr__(
            self,
            "axes",
            tuple(
                a if isinstance(a, AxisConfig) else AxisConfig.from_dict(a)
                for a in self.axes
            ),
        )


@dataclass(frozen=True, slots=True)
class ChainConfig:
    start: str
    end: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ChainConfig":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PartConfig:
    sided: bool = False
    landmarks: dict[str, LandmarkConfig] = field(default_factory=dict)
    segments: dict[str, SegmentConfig] = field(default_factory=dict)
    chains: dict[str, ChainConfig] = field(default_factory=dict)

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
