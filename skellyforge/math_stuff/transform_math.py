import math
from dataclasses import dataclass

from skellyforge.math_stuff.rotation_quaternion import RotationQuaternion


@dataclass(slots=True, frozen=True)
class Position:
    x: float
    y: float
    z: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in [self.x, self.y, self.z]):
            raise ValueError("Position be 3 finite numbers")


class Transform:
    position: Position
    quaternion: RotationQuaternion
