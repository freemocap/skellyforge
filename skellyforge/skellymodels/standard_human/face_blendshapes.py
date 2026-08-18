"""The face: 52 ARKit blend shapes, hydrated later from face tracking.

A separate object from the skeleton - the eyes / ears / nose are LANDMARKS on
the skull (see axial.yaml's head), while the face is a blendshape object that
rides alongside and is zero until face-tracking data hydrates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.skellymodels.standard_human.config_types import FaceConfig
from skellyforge.skellymodels.standard_human.human_skeleton import load_config


@dataclass(frozen=True, slots=True)
class FaceBlendShapes:
    name: str
    blendshapes: dict[str, float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FaceBlendShapes":
        path = Path(path)
        config = FaceConfig.from_dict(
            load_config(yaml.safe_load(path.read_text()), path.parent)
        )
        return cls(name=config.name, blendshapes=config.blendshapes)
