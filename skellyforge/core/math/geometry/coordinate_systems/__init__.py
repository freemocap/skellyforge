"""Coordinate-system conventions and conversion between them.

The codebase is authored in one frame - Blender (+X right, +Y forward, +Z up) - declared
as the default in definitions/coordinate_systems/coordinate_systems.yaml. Other frames
(VRM/glTF, ROS, ISB, Unreal, Unity, and any a user defines) are named in that same file
and entered only at I/O boundaries, through CoordinateSystemTransform.
"""

from skellyforge.core.math.geometry.coordinate_systems.anatomical_direction import (
    AnatomicalDirection,
    parse_anatomical_direction,
)
from skellyforge.core.math.geometry.coordinate_systems.coordinate_system_convention import (
    CoordinateSystemConvention,
)
from skellyforge.core.math.geometry.coordinate_systems.coordinate_system_registry import (
    CoordinateSystemRegistry,
)
from skellyforge.core.math.geometry.coordinate_systems.coordinate_system_transform import (
    CoordinateSystemTransform,
    conversion_matrix,
)
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness

__all__ = [
    "AnatomicalDirection",
    "CoordinateSystemConvention",
    "CoordinateSystemRegistry",
    "CoordinateSystemTransform",
    "Handedness",
    "conversion_matrix",
    "parse_anatomical_direction",
]
