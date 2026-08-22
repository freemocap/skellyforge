"""Gram-Schmidt construction of an orthonormal reference frame from three named points.

A `ReferenceFrameDefinition` names:
    * the point that is the origin of the frame,
    * a point that lies EXACTLY along a signed axis (e.g. `SpatialAxis.NEGATIVE_Y`),
    * a point that lies APPROXIMATELY along a second signed axis,
    * the handedness of the resulting coordinate system (right-handed by default).

`calculate_orthonormal_basis` turns that definition plus a mapping of named points into
an `OrthonormalBasis`: an orthonormal (x, y, z) triad plus the origin, which transforms
points between the world frame and the newly defined local frame.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_SINE_BETWEEN_DEFINING_VECTORS,
)
from skellyforge.core.math.geometry.orthonormal_basis.orthonormal_basis import OrthonormalBasis
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point, UnitVector


def calculate_orthonormal_basis(
    *,
    points: Mapping[str, Point],
    definition: ReferenceFrameDefinition,
) -> OrthonormalBasis:
    """Build an orthonormal basis by Gram-Schmidt from three named points.

    The primary axis points exactly from the origin point to the primary point, reversed
    if the primary signed axis is negative. The secondary axis is the component of the
    origin-to-secondary-point displacement perpendicular to the primary axis, likewise
    reversed for a negative signed axis. The tertiary axis is their cross product, signed
    so that the triad has the requested handedness.

    Args:
        points: mapping of point name -> world location. All points named by the
            definition must share identical shapes.
        definition: the origin/primary/secondary/handedness spec of the frame.

    Returns:
        The `OrthonormalBasis` described by `definition`.

    Raises:
        KeyError: a point named by the definition is missing from `points`.
        ValueError: the definition is underspecified, the defining points have mismatched
            shapes, two of them are coincident, or the defining displacements are
            collinear.
    """
    secondary_axis = definition.secondary_axis
    secondary_point_name = definition.secondary_point_name
    if secondary_axis is None or secondary_point_name is None:
        raise ValueError(
            "Cannot build an orthonormal basis from an underspecified reference frame "
            f"definition ({definition}) - `{definition.primary_point_name}` fixes only "
            f"the {definition.primary_axis.name} direction, leaving roll about it free. "
            "Use `direction_along` for the direction, and resolve roll separately."
        )

    required_names = (
        definition.origin_point_name,
        definition.primary_point_name,
        secondary_point_name,
    )
    missing_names = [name for name in required_names if name not in points]
    if missing_names:
        raise KeyError(
            f"Points {missing_names} required by the reference frame definition are "
            f"missing - available points: {sorted(points.keys())}"
        )

    origin, primary_point, secondary_point = (points[name] for name in required_names)
    if not (origin.array.shape == primary_point.array.shape == secondary_point.array.shape):
        raise ValueError(
            "Defining points must all have the same shape - got "
            f"{definition.origin_point_name}: {origin.array.shape}, "
            f"{definition.primary_point_name}: {primary_point.array.shape}, "
            f"{secondary_point_name}: {secondary_point.array.shape}"
        )

    primary_direction = direction_along(
        axis=definition.primary_axis,
        displacement=primary_point - origin,
        description=(
            f"the primary axis displacement (`{definition.origin_point_name}` -> "
            f"`{definition.primary_point_name}`)"
        ),
    )
    approximate_secondary_direction = direction_along(
        axis=secondary_axis,
        displacement=secondary_point - origin,
        description=(
            f"the approximate secondary axis displacement "
            f"(`{definition.origin_point_name}` -> `{secondary_point_name}`)"
        ),
    )

    # Gram-Schmidt: strip the primary-axis component out of the approximate secondary
    # direction. The residual's length is the sine of the angle between the two.
    projection_lengths = approximate_secondary_direction.dot(other=primary_direction)
    residual: Displacement = approximate_secondary_direction.as_displacement() - (
        primary_direction.scaled_by(factors=projection_lengths)
    )
    residual_lengths = residual.norm()
    if np.any(residual_lengths < MINIMUM_SINE_BETWEEN_DEFINING_VECTORS):
        raise ValueError(
            f"The primary displacement (`{definition.origin_point_name}` -> "
            f"`{definition.primary_point_name}`) and the approximate secondary "
            f"displacement (`{definition.origin_point_name}` -> "
            f"`{secondary_point_name}`) are collinear - smallest sine of the "
            f"angle between them is {float(residual_lengths.min()):.3e} < "
            f"{MINIMUM_SINE_BETWEEN_DEFINING_VECTORS:.1e}"
        )
    secondary_direction = residual.normalized(
        description="the secondary axis orthogonalized against the primary axis"
    )

    # The two directions are perpendicular, so their cross product is a unit vector along
    # the tertiary axis. The cyclic sign says which end of that axis it landed on, and the
    # handedness flips it once more for a left-handed triad.
    tertiary_sign = definition.handedness.value * definition.primary_axis.cyclic_sign_toward(
        secondary_axis
    )
    tertiary_direction = _signed(
        direction=primary_direction.cross(other=secondary_direction), sign=tertiary_sign
    )

    directions_by_index: dict[int, UnitVector] = {
        definition.primary_axis.index: primary_direction,
        secondary_axis.index: secondary_direction,
        definition.tertiary_axis.index: tertiary_direction,
    }
    return OrthonormalBasis(
        origin=origin,
        x_axis=directions_by_index[SpatialAxis.X.index],
        y_axis=directions_by_index[SpatialAxis.Y.index],
        z_axis=directions_by_index[SpatialAxis.Z.index],
        handedness=definition.handedness,
    )


def direction_along(
    *, axis: SpatialAxis, displacement: Displacement, description: str
) -> UnitVector:
    """The unit vector that puts `displacement` on the signed half-axis named by `axis`.

    For a negative axis the direction points opposite the named point, which is exactly
    what makes that point land on the negative half of its declared axis.
    """
    return _signed(direction=displacement.normalized(description=description), sign=axis.sign)


def _signed(*, direction: UnitVector, sign: int) -> UnitVector:
    """`direction` as-is for a positive sign, reversed for a negative one."""
    return direction if sign > 0 else -direction
