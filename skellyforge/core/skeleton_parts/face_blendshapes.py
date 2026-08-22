"""The 52 ARKit blendshapes that drive the standard-human face.

The face is not part of the segment/landmark skeleton - it is a set of named scalar
expressions whose weights animate the facial mesh each frame. This module holds the names
(and, later, whatever per-blendshape data the driver needs). Driving them - mapping
observed keypoints to weights - comes later, and is deliberately not here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

BlendshapeName = str


@dataclass(frozen=True, slots=True, eq=False)
class FaceBlendShapes:
    """One face's set of blendshape names, loaded from YAML.

    Attributes:
        name: what this face is called.
        blendshape_names: the named scalar expressions, in authoring order.
    """

    name: str
    blendshape_names: tuple[BlendshapeName, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("face blendshape set needs a name")
        if not self.blendshape_names:
            raise ValueError(f"face {self.name!r} needs at least one blendshape")
        duplicates = sorted(
            {
                name
                for name in self.blendshape_names
                if self.blendshape_names.count(name) > 1
            }
        )
        if duplicates:
            raise ValueError(
                f"face {self.name!r}: blendshape names must be unique - repeated: "
                f"{duplicates}"
            )

    def __len__(self) -> int:
        """How many blendshapes this face carries."""
        return len(self.blendshape_names)

    @classmethod
    def from_yaml(cls, *, path: Path) -> FaceBlendShapes:
        """Load a face blendshape set from its YAML file."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"{path} must parse to a mapping, got {type(document).__name__}")
        names = document.get("blendshapes")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
            raise ValueError(f"{path} needs a non-empty `blendshapes` list")
        return cls(
            name=str(document.get("name", path.stem)),
            blendshape_names=tuple(str(name) for name in names),
        )
