"""The T-pose: the static reference geometry every live pose is measured against.

SegmentTposeGeometry is one segment's rest geometry (world origin, rest basis,
length). StandardHumanTPose is the whole human's T-pose: per-segment geometry
plus per-landmark rest-world positions. Built once from a HumanSkeleton by
build_standard_human_tpose (kinematics.tpose); the orientation solver measures
live poses against it (identity == T-pose).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.quaternion_math import RotationQuaternion


@dataclass(frozen=True, slots=True)
class SegmentTposeGeometry:
    """One segment's T-pose geometry: where it sits, how it's oriented, how long."""

    origin: NDArray[np.float64]  # (3,) world rest origin
    basis: NDArray[np.float64]   # (3,3) rows [x-hat, y-hat, z-hat] of the rest frame
    length: float

    @property
    def rotation_quaternion(self) -> NDArray[np.float64]:
        """The rest frame as a wxyz quaternion (a view of the basis)."""
        q = RotationQuaternion.from_rotation_matrix(self.basis)
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class StandardHumanTPose:
    """The whole human's T-pose: per-segment geometry + rest landmark positions."""

    segments: dict[str, SegmentTposeGeometry]
    landmarks: dict[str, NDArray[np.float64]]
