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

"""ARKit blendshape channel declarations for the standard human model.

The 52 ARKit blendshapes are the de facto standard for face tracking data.
They are declared here so the frame's model can enumerate them
and downstream adapters can map them to target expression systems (VRM,
VRChat, Unreal Metahuman, etc.).

All channels are declared but **undriven** — values are ``NaN`` on the
stream until SkellyTracker's face tracking is wired into the pipeline.
This follows the same positions-first / rotations-NaN pattern used for
body data.

VRM expression mapping is noted per-channel where a direct equivalent
exists. The Phase-3 VMC adapter will use this mapping at emit time.
"""

from enum import Enum


class BlendShapeChannel(str, Enum):
    """Standard ARKit blendshape channel names.

    Enum values are the Apple ARKit names (camelCase), matching the
    names used by live link face tracking and VTuber software.
    """

    # ── Brow ───────────────────────────────────────────────────────────
    BROW_DOWN_LEFT = "browDownLeft"
    BROW_DOWN_RIGHT = "browDownRight"
    BROW_INNER_UP = "browInnerUp"
    BROW_OUTER_UP_LEFT = "browOuterUpLeft"
    BROW_OUTER_UP_RIGHT = "browOuterUpRight"

    # ── Eye ────────────────────────────────────────────────────────────
    EYE_BLINK_LEFT = "eyeBlinkLeft"
    EYE_BLINK_RIGHT = "eyeBlinkRight"
    EYE_LOOK_DOWN_LEFT = "eyeLookDownLeft"
    EYE_LOOK_DOWN_RIGHT = "eyeLookDownRight"
    EYE_LOOK_IN_LEFT = "eyeLookInLeft"
    EYE_LOOK_IN_RIGHT = "eyeLookInRight"
    EYE_LOOK_OUT_LEFT = "eyeLookOutLeft"
    EYE_LOOK_OUT_RIGHT = "eyeLookOutRight"
    EYE_LOOK_UP_LEFT = "eyeLookUpLeft"
    EYE_LOOK_UP_RIGHT = "eyeLookUpRight"
    EYE_SQUINT_LEFT = "eyeSquintLeft"
    EYE_SQUINT_RIGHT = "eyeSquintRight"
    EYE_WIDE_LEFT = "eyeWideLeft"
    EYE_WIDE_RIGHT = "eyeWideRight"

    # ── Cheek ──────────────────────────────────────────────────────────
    CHEEK_PUFF = "cheekPuff"
    CHEEK_SQUINT_LEFT = "cheekSquintLeft"
    CHEEK_SQUINT_RIGHT = "cheekSquintRight"

    # ── Nose ───────────────────────────────────────────────────────────
    NOSE_SNEER_LEFT = "noseSneerLeft"
    NOSE_SNEER_RIGHT = "noseSneerRight"

    # ── Mouth (jaw) ────────────────────────────────────────────────────
    MOUTH_CLOSE = "mouthClose"
    MOUTH_DIMPLE_LEFT = "mouthDimpleLeft"
    MOUTH_DIMPLE_RIGHT = "mouthDimpleRight"
    MOUTH_FROWN_LEFT = "mouthFrownLeft"
    MOUTH_FROWN_RIGHT = "mouthFrownRight"
    MOUTH_FUNNEL = "mouthFunnel"
    MOUTH_LEFT = "mouthLeft"
    MOUTH_LOWER_DOWN_LEFT = "mouthLowerDownLeft"
    MOUTH_LOWER_DOWN_RIGHT = "mouthLowerDownRight"
    MOUTH_PRESS_LEFT = "mouthPressLeft"
    MOUTH_PRESS_RIGHT = "mouthPressRight"
    MOUTH_PUCKER = "mouthPucker"
    MOUTH_RIGHT = "mouthRight"
    MOUTH_ROLL_LOWER = "mouthRollLower"
    MOUTH_ROLL_UPPER = "mouthRollUpper"
    MOUTH_SHRUG_LOWER = "mouthShrugLower"
    MOUTH_SHRUG_UPPER = "mouthShrugUpper"
    MOUTH_SMILE_LEFT = "mouthSmileLeft"
    MOUTH_SMILE_RIGHT = "mouthSmileRight"
    MOUTH_STRETCH_LEFT = "mouthStretchLeft"
    MOUTH_STRETCH_RIGHT = "mouthStretchRight"
    MOUTH_UPPER_UP_LEFT = "mouthUpperUpLeft"
    MOUTH_UPPER_UP_RIGHT = "mouthUpperUpRight"

    # ── Tongue ─────────────────────────────────────────────────────────
    TONGUE_OUT = "tongueOut"

    # ── Jaw ────────────────────────────────────────────────────────────
    JAW_FORWARD = "jawForward"
    JAW_LEFT = "jawLeft"
    JAW_OPEN = "jawOpen"
    JAW_RIGHT = "jawRight"


# ── Channel metadata ───────────────────────────────────────────────────


def get_blendshape_count() -> int:
    """Return the number of declared blendshape channels (one per ``BlendShapeChannel`` member)."""
    return len(BlendShapeChannel)


def get_blendshape_names() -> list[str]:
    """Return all blendshape channel names in declaration order."""
    return [channel.value for channel in BlendShapeChannel]


# ── VRM expression mapping ─────────────────────────────────────────────
#
# When face tracking is wired, ARKit weights map to VRM expressions.
# This is the reference mapping; the VMC adapter uses it at emit time.
#
# VRM 0.x preset → ARKit channels that drive it (dominant one(s) first)
#
# VRM 1.0 preset names are noted alongside for when the adapter targets
# VRM 1.0 consumers directly.

VRM_EXPRESSION_ARKIT_MAPPING: dict[str, list[str]] = {
    # VRM 0.x preset        ARKit channels
    # (also VRM 1.0 name)
    "joy": [                 # VRM 1.0: happy
        "mouthSmileLeft",
        "mouthSmileRight",
    ],
    "angry": [               # VRM 1.0: angry
        "browDownLeft",
        "browDownRight",
        "mouthFrownLeft",
        "mouthFrownRight",
    ],
    "sorrow": [              # VRM 1.0: sad
        "browInnerUp",
        "mouthFrownLeft",
        "mouthFrownRight",
    ],
    "fun": [                 # VRM 1.0: relaxed
        "mouthPucker",
    ],
    "surprised": [           # VRM 1.0 only
        "browInnerUp",
        "eyeWideLeft",
        "eyeWideRight",
        "jawOpen",
    ],
    "a": [                   # VRM 1.0: aa
        "jawOpen",
    ],
    "i": [                   # VRM 1.0: ih
        "mouthStretchLeft",
        "mouthStretchRight",
    ],
    "u": [                   # VRM 1.0: ou
        "mouthFunnel",
    ],
    "e": [                   # VRM 1.0: ee
        "mouthSmileLeft",
        "mouthSmileRight",
    ],
    "o": [                   # VRM 1.0: oh
        "mouthFunnel",
        "jawOpen",
    ],
    "blink_l": [             # VRM 1.0: blinkLeft
        "eyeBlinkLeft",
    ],
    "blink_r": [             # VRM 1.0: blinkRight
        "eyeBlinkRight",
    ],
    "lookUp": [              # VRM 1.0: lookUp
        "eyeLookUpLeft",
        "eyeLookUpRight",
    ],
    "lookDown": [            # VRM 1.0: lookDown
        "eyeLookDownLeft",
        "eyeLookDownRight",
    ],
    "lookLeft": [            # VRM 1.0: lookLeft
        "eyeLookOutLeft",
        "eyeLookInRight",
    ],
    "lookRight": [           # VRM 1.0: lookRight
        "eyeLookOutRight",
        "eyeLookInLeft",
    ],
}

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
