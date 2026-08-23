"""Per-segment inertia tensors, from de Leva radii of gyration and segment orientation.

de Leva gives each segment's mass distribution as three radii of gyration, each a fraction
of segment length, about the segment's principal axes: sagittal (anterior-posterior),
transverse (medio-lateral) and longitudinal (along the bone). This module turns those into
a 3x3 world-frame inertia tensor about the segment's center of mass, by building the
principal frame from the segment's long axis and the world forward direction.
"""

from __future__ import annotations

import numpy as np

from skellyforge.core.biomechanics.anthropometric_parameters import RadiiOfGyration
from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.type_overloads import FloatArray

_FORWARD: FloatArray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_UP: FloatArray = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def segment_inertia_tensor(
    *,
    mass: float,
    radii: RadiiOfGyration,
    proximal: FloatArray,
    distal: FloatArray,
) -> FloatArray:
    """The inertia tensor of one segment about its COM, in the world frame.

    Args:
        mass: the segment's mass.
        radii: radii of gyration about the principal axes, fractions of segment length.
        proximal: the world position of the segment's proximal (long-axis) end.
        distal: the world position of the segment's distal end.

    Returns:
        A 3x3 symmetric inertia tensor about the segment's center of mass.

    Raises:
        ValueError: the proximal and distal anchors coincide, so the long axis is
            undefined; or the segment's long axis is parallel to the world forward
            direction, so no sagittal axis can be projected from it.
    """
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")

    long_vector = np.asarray(distal, dtype=np.float64) - np.asarray(
        proximal, dtype=np.float64
    )
    length = float(np.linalg.norm(long_vector))
    if length < MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"segment has zero length - proximal {proximal!r} and distal {distal!r} coincide"
        )
    longitudinal = long_vector / length

    sagittal = _project_perpendicular(vector=_FORWARD, axis=longitudinal)
    sagittal_norm = float(np.linalg.norm(sagittal))
    if sagittal_norm < MINIMUM_VECTOR_NORM:
        sagittal = _project_perpendicular(vector=_UP, axis=longitudinal)
        sagittal_norm = float(np.linalg.norm(sagittal))
        if sagittal_norm < MINIMUM_VECTOR_NORM:
            raise ValueError(
                f"segment long axis {longitudinal!r} is parallel to both the forward and "
                "up directions - cannot build a principal frame"
            )
    sagittal_axis = sagittal / sagittal_norm

    transverse_axis = np.cross(sagittal_axis, longitudinal)

    # Columns are the principal axes in (transverse, sagittal, longitudinal) order, which
    # is the order the radii are stored in.
    rotation = np.stack([transverse_axis, sagittal_axis, longitudinal], axis=1)
    principal = (
        mass
        * length**2
        * np.diag(
            [radii.transverse**2, radii.sagittal**2, radii.longitudinal**2]
        )
    )
    return rotation @ principal @ rotation.T


def _project_perpendicular(*, vector: FloatArray, axis: FloatArray) -> FloatArray:
    """The component of vector perpendicular to axis."""
    return vector - float(np.dot(vector, axis)) * axis
