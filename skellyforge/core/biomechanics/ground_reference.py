"""Ground-reference quantities: center of pressure, extrapolated COM, moment pivot.

These are the balance measures computed on the ground plane (z = 0 in the Blender
convention). They are pure functions of the whole-body center of mass, its velocity, and
the ground reaction force and moment - all of which the rest of this package derives or
the caller supplies from a force plate.

Formulas (Hof 2008 for the extrapolated center of mass; Popovic, Goswami & Herr 2005 for
the centroidal moment pivot):

  XCoM = CoM + v_CoM / omega0,   omega0 = sqrt(g / h),  h = CoM height above the ground

  CoP  = (-M_y / F_z,  M_x / F_z, 0),  with the moment measured at the contact surface

  CMP  = ( CoM_x - (F_x / F_z) h,  CoM_y - (F_y / F_z) h, 0 )

The CMP is the point where a line parallel to the ground reaction force, through the CoM,
meets the ground: the pivot about which the force would generate no horizontal moment
about the CoM.
"""

from __future__ import annotations

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.type_overloads import FloatArray

# Blender convention: +z up, ground at z = 0. Gravity points down, in mm/s^2.
GRAVITY_ACCELERATION: FloatArray = np.array([0.0, 0.0, -9810.0], dtype=np.float64)


def extrapolated_center_of_mass(
    *,
    com: FloatArray,
    com_velocity: FloatArray,
    gravity: FloatArray | None = None,
) -> FloatArray:
    """The extrapolated center of mass (XCoM), on the ground plane.

    The XCoM is where the CoM would be if it kept moving at its current velocity, under
    the inverted-pendulum natural frequency of a pendulum as tall as the CoM. It is the
    single point that predicts whether the CoM is being brought back over the base of
    support: a still body has XCoM == CoM_ground; a body falling forward has XCoM ahead
    of the CoP.

    Args:
        com: the whole-body center of mass (world, +z up, ground at z = 0).
        com_velocity: the CoM velocity (world, same units per second as the skeleton).
        gravity: the gravity vector; defaults to -z at 9810 mm/s^2.

    Raises:
        ValueError: the CoM is at or below the ground, so there is no pendulum height.
    """
    gravity = GRAVITY_ACCELERATION if gravity is None else np.asarray(gravity, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)
    velocity = np.asarray(com_velocity, dtype=np.float64)

    height = float(com[2])
    if height <= MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"the CoM must be above the ground to extrapolate - got height {height}"
        )

    gravity_magnitude = float(np.linalg.norm(gravity))
    omega0 = np.sqrt(gravity_magnitude / height)
    extrapolated = com + velocity / omega0
    extrapolated[2] = 0.0
    return extrapolated


def center_of_pressure(*, force: FloatArray, moment: FloatArray) -> FloatArray:
    """The center of pressure (CoP) on the ground, from a force plate.

    Args:
        force: the ground reaction force (F_x, F_y, F_z) at the contact surface, F_z up.
        moment: the moment (M_x, M_y, M_z) measured at the contact surface.

    Raises:
        ValueError: the vertical force is zero, so the CoP is undefined.
    """
    force = np.asarray(force, dtype=np.float64)
    moment = np.asarray(moment, dtype=np.float64)
    vertical_force = float(force[2])
    if abs(vertical_force) <= MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"vertical ground reaction force must be nonzero to locate the CoP - got "
            f"{vertical_force}"
        )
    return np.array(
        [-moment[1] / vertical_force, moment[0] / vertical_force, 0.0], dtype=np.float64
    )


def centroidal_moment_pivot(
    *, force: FloatArray, center_of_mass: FloatArray
) -> FloatArray:
    """The centroidal moment pivot (CMP), on the ground plane.

    The CMP is where a line parallel to the ground reaction force, through the CoM, meets
    the ground - equivalently, the point on the ground where the force would act to
    generate no horizontal moment about the CoM.

    Args:
        force: the ground reaction force (F_x, F_y, F_z), F_z up.
        center_of_mass: the whole-body center of mass (world, +z up).

    Raises:
        ValueError: the vertical force is zero, so the pivot is undefined.
    """
    force = np.asarray(force, dtype=np.float64)
    com = np.asarray(center_of_mass, dtype=np.float64)
    vertical_force = float(force[2])
    if abs(vertical_force) <= MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"vertical ground reaction force must be nonzero to locate the CMP - got "
            f"{vertical_force}"
        )
    height = float(com[2])
    return np.array(
        [
            com[0] - (force[0] / vertical_force) * height,
            com[1] - (force[1] / vertical_force) * height,
            0.0,
        ],
        dtype=np.float64,
    )
