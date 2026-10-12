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


def segment_inertia_tensors(*, masses: FloatArray, radii: FloatArray,
                            proximal: FloatArray, distal: FloatArray) -> FloatArray:
    """Matching-row inertia tensors; radii columns are transverse/sagittal/longitudinal."""
    masses = np.asarray(masses, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    proximal = np.asarray(proximal, dtype=np.float64)
    distal = np.asarray(distal, dtype=np.float64)
    if masses.ndim != 1 or any(array.shape != (len(masses), 3) for array in (radii, proximal, distal)):
        raise ValueError('Expected (N,) masses and matching (N, 3) radii and anchors')
    if np.any(masses <= 0.0):
        raise ValueError('mass must be positive')
    long_vectors = distal - proximal
    lengths = np.linalg.norm(long_vectors, axis=1)
    if np.any(lengths < MINIMUM_VECTOR_NORM):
        raise ValueError('segment has zero length - proximal and distal coincide')
    longitudinal = long_vectors / lengths[:, None]
    sagittal = _FORWARD - longitudinal[:, 1, None] * longitudinal
    sagittal_norms = np.linalg.norm(sagittal, axis=1)
    fallback = sagittal_norms < MINIMUM_VECTOR_NORM
    sagittal[fallback] = _UP - longitudinal[fallback, 2, None] * longitudinal[fallback]
    sagittal_norms[fallback] = np.linalg.norm(sagittal[fallback], axis=1)
    if np.any(sagittal_norms < MINIMUM_VECTOR_NORM):
        raise ValueError('Cannot build a principal frame')
    sagittal_axes = sagittal / sagittal_norms[:, None]
    transverse = np.cross(sagittal_axes, longitudinal)
    rotations = np.stack([transverse, sagittal_axes, longitudinal], axis=2)
    principal = np.zeros((len(masses), 3, 3), dtype=np.float64)
    principal[:, np.arange(3), np.arange(3)] = masses[:, None] * lengths[:, None] ** 2 * radii ** 2
    return rotations @ principal @ rotations.transpose(0, 2, 1)


def _project_perpendicular(*, vector: FloatArray, axis: FloatArray) -> FloatArray:
    """The component of vector perpendicular to axis."""
    return vector - float(np.dot(vector, axis)) * axis
