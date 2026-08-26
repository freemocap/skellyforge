"""Closed-form two-bone inverse kinematics: root, two lengths, a target.

The analytic limb solver - the exact triangle that puts an elbow where a
two-segment chain can reach a target. Closed form, no iteration: law of
cosines places the middle joint on the pole side of the root->target axis,
and the end lands exactly on the target whenever the target is reachable.

Reachability is enforced loudly. A target farther than ``upper + lower`` or
closer than ``|upper - lower|`` is not a clamped near-miss here - it is an
impossible request, and it raises naming the distances so the caller can
decide what to do about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.math.kinematics.coordinate_frame_ops import default_perpendicular
from skellyforge.type_overloads import FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class TwoBoneIkSolution:
    """The solved limb geometry for one target.

    Attributes:
        middle_joint: where the elbow/knee lands (on the pole side).
        end_joint: the solved end position - exactly the target when reachable.
        reach_distance: how far the target sits from the root.
        maximum_reach: ``upper_length + lower_length``, for context in reports.
    """

    middle_joint: Point
    end_joint: Point
    reach_distance: float
    maximum_reach: float


def raise_unless_lengths_are_valid(
    *, upper_length: float, lower_length: float, solver_name: str
) -> None:
    """Shared length validation for both IK solvers."""
    for label, length in (("upper", upper_length), ("lower", lower_length)):
        if not np.isfinite(length) or length <= MINIMUM_VECTOR_NORM:
            raise ValueError(
                f"{solver_name}: {label} length must be a positive, finite number "
                f"- got {length}"
            )


def solve_two_bone_ik(
    *,
    root_position: Point,
    target: Point,
    upper_length: float,
    lower_length: float,
    pole_hint: FloatArray | None = None,
) -> TwoBoneIkSolution:
    """Solve the two-bone triangle placing `target` at the chain's end.

    Args:
        root_position: the proximal joint's world position.
        target: where the distal end should land.
        upper_length: root -> middle bone length.
        lower_length: middle -> end bone length.
        pole_hint: world direction indicating which SIDE the middle joint
            should bend toward (elbow-down vs elbow-up). Projected
            perpendicular to the root->target axis. Defaults to a
            deterministic perpendicular of the axis.

    Returns:
        The solved positions. ``end_joint == target`` by construction.

    Raises:
        ValueError: either length is non-positive; the target coincides with
            the root; or the target is unreachable - farther than the summed
            lengths or closer than their difference (a fully-folded limb).
    """
    raise_unless_lengths_are_valid(
        upper_length=upper_length,
        lower_length=lower_length,
        solver_name="solve_two_bone_ik",
    )
    if pole_hint is not None and float(np.linalg.norm(np.asarray(pole_hint))) < MINIMUM_VECTOR_NORM:
        raise ValueError("pole_hint must be a nonzero direction when given")

    delta = target.array - root_position.array
    reach = float(np.linalg.norm(delta))
    total_reach = upper_length + lower_length
    minimum_reach = abs(upper_length - lower_length)

    if reach < MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"two-bone IK: the target coincides with the root "
            f"({np.asarray(root_position.array).tolist()}) - no limb direction exists"
        )
    if reach > total_reach:
        raise ValueError(
            f"two-bone IK: target unreachable - distance {reach:.3f} exceeds the "
            f"limb's full reach {total_reach:.3f} "
            f"(upper {upper_length:.3f} + lower {lower_length:.3f})"
        )
    if reach < minimum_reach:
        raise ValueError(
            f"two-bone IK: target unreachable - distance {reach:.3f} is closer than "
            f"the folded limb allows ({minimum_reach:.3f} = |{upper_length:.3f} - "
            f"{lower_length:.3f}|)"
        )

    axis = delta / reach
    if pole_hint is not None:
        hint = np.asarray(pole_hint, dtype=np.float64)
        hint = hint / np.linalg.norm(hint)
        perpendicular = hint - float(np.dot(hint, axis)) * axis
        perpendicular_norm = float(np.linalg.norm(perpendicular))
        if perpendicular_norm < MINIMUM_VECTOR_NORM:
            # A pole parallel to the axis carries no side information; fall
            # back to the deterministic perpendicular rather than dividing
            # into rounding error.
            perpendicular = default_perpendicular(direction=axis)
        else:
            perpendicular = perpendicular / perpendicular_norm
    else:
        perpendicular = default_perpendicular(direction=axis)

    # Law of cosines: interior angle at the root between the axis and the
    # upper bone. At exact reach boundaries this is ±1 and the limb is
    # straight - the perpendicular term below drops to zero on its own.
    cosine_at_root = (
        upper_length**2 + reach**2 - lower_length**2
    ) / (2.0 * upper_length * reach)
    cosine_at_root = float(np.clip(cosine_at_root, -1.0, 1.0))
    along_axis = upper_length * cosine_at_root
    toward_pole = float(np.sqrt(max(upper_length**2 - along_axis**2, 0.0)))

    middle_array = root_position.array + axis * along_axis + perpendicular * toward_pole
    return TwoBoneIkSolution(
        middle_joint=Point.from_prevalidated_array(array=middle_array),
        end_joint=Point.from_prevalidated_array(array=target.array.copy()),
        reach_distance=reach,
        maximum_reach=total_reach,
    )
