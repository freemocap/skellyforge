"""FABRIK inverse kinematics: forward-backward reaching for N-segment chains.

FABRIK (Forward And Backward Reaching Inverse Kinematics) walks a joint chain
toward a target: a backward pass pins the end on the target and re-spaces the
joints at their bone lengths back to the root, a forward pass pins the root and
re-spaces forward. Each pass preserves every bone length exactly; iterating the
pair converges geometrically for reachable targets.

Convergence is a contract, not a hope: a chain that has not met the tolerance
within ``max_iterations`` raises, naming its residual - silent near-misses are
how IK bugs hide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.type_overloads import FloatArray

# A convergence criterion for an iterative solver, in millimetres - a different
# species from the vector-degeneracy guards in numeric_tolerances.py, so it
# lives beside the algorithm whose meaning it carries.
DEFAULT_FABRIK_TOLERANCE_MM: Final[float] = 0.01
DEFAULT_FABRIK_MAX_ITERATIONS: Final[int] = 100


@dataclass(frozen=True, slots=True, eq=False)
class FabrikSolution:
    """The solved chain plus the report a caller needs to trust it.

    Attributes:
        joint_positions: root -> end positions after solving; every consecutive
            pair is exactly one bone length apart.
        iterations_used: forward/backward passes performed.
        final_residual: distance from the solved end to the target, in the
            position units of the input (millimetres here).
        converged: whether `final_residual` met the tolerance. Always true
            unless the solve was clamped out-of-reach with permission.
    """

    joint_positions: tuple[Point, ...]
    iterations_used: int
    final_residual: float
    converged: bool


def _raise_unless_chain_is_solvable(*, joints: list[FloatArray]) -> list[float]:
    if len(joints) < 2:
        raise ValueError(
            f"FABRIK needs at least two joints (one bone) - got {len(joints)}"
        )
    lengths: list[float] = []
    for index in range(len(joints) - 1):
        bone = joints[index + 1] - joints[index]
        length = float(np.linalg.norm(bone))
        if not np.isfinite(length) or length <= MINIMUM_VECTOR_NORM:
            raise ValueError(
                f"FABRIK: bone {index}->{index + 1} has degenerate length "
                f"{length:.3e} - bones must be positive and finite"
            )
        lengths.append(length)
    return lengths


def solve_fabrik(
    *,
    chain_positions: Sequence[Point],
    target: Point,
    tolerance_mm: float = DEFAULT_FABRIK_TOLERANCE_MM,
    max_iterations: int = DEFAULT_FABRIK_MAX_ITERATIONS,
    allow_out_of_reach: bool = False,
) -> FabrikSolution:
    """Iterate the chain toward `target`, preserving every bone length.

    Args:
        chain_positions: the current joint positions, root first. Their
            spacings define the bone lengths, which are preserved exactly.
        target: where the end joint should go.
        tolerance_mm: convergence threshold on end-to-target distance.
        max_iterations: hard cap on passes; exhaustion raises rather than
            returning an unreported near-miss.
        allow_out_of_reach: when False (default), a target beyond the chain's
            total reach raises immediately. When True, the chain instead
            straightens fully toward the target - the honest best it can do -
            and reports ``converged=False`` with the remaining residual.

    Returns:
        The solved chain with its diagnostics.

    Raises:
        ValueError: fewer than two joints, a degenerate bone length, a
            non-positive tolerance, iteration exhaustion without convergence,
            or an out-of-reach target while ``allow_out_of_reach`` is False.
    """
    if tolerance_mm <= 0.0:
        raise ValueError(f"tolerance must be positive - got {tolerance_mm}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1 - got {max_iterations}")

    joints = [point.array.astype(np.float64).copy() for point in chain_positions]
    lengths = _raise_unless_chain_is_solvable(joints=joints)

    target_array = np.asarray(target.array, dtype=np.float64)
    total_reach = float(sum(lengths))
    root_target_distance = float(np.linalg.norm(target_array - joints[0]))

    if root_target_distance > total_reach and not allow_out_of_reach:
        raise ValueError(
            f"FABRIK: target unreachable - distance {root_target_distance:.3f} "
            f"exceeds the chain's full reach {total_reach:.3f}. Pass "
            "allow_out_of_reach=True to straighten toward it instead."
        )

    def backward_pass(joints: list[FloatArray]) -> list[FloatArray]:
        joints[-1] = target_array.copy()
        for index in range(len(lengths) - 1, -1, -1):
            direction = joints[index] - joints[index + 1]
            norm = float(np.linalg.norm(direction))
            if norm < MINIMUM_VECTOR_NORM:
                # Coincident joints mid-iteration: nudge deterministically
                # along the incoming bone's direction rather than dividing by
                # zero. Cannot occur from valid non-degenerate inputs except
                # through catastrophic cancellation, but the guard keeps the
                # solver total.
                direction = joints[index + 2] - joints[index + 1] if index + 2 < len(joints) else np.array([1.0, 0.0, 0.0])
                norm = float(np.linalg.norm(direction))
                if norm < MINIMUM_VECTOR_NORM:
                    direction = np.array([1.0, 0.0, 0.0])
                    norm = 1.0
            joints[index] = joints[index + 1] + direction / norm * lengths[index]
        return joints

    def forward_pass(joints: list[FloatArray]) -> list[FloatArray]:
        joints[0] = np.asarray(chain_positions[0].array, dtype=np.float64).copy()
        for index in range(len(lengths)):
            direction = joints[index + 1] - joints[index]
            norm = float(np.linalg.norm(direction))
            if norm < MINIMUM_VECTOR_NORM:
                direction = joints[index + 1] - joints[index - 1] if index > 0 else np.array([0.0, 0.0, 1.0])
                norm = float(np.linalg.norm(direction))
                if norm < MINIMUM_VECTOR_NORM:
                    direction = np.array([0.0, 0.0, 1.0])
                    norm = 1.0
            joints[index + 1] = joints[index] + direction / norm * lengths[index]
        return joints

    if root_target_distance > total_reach:
        # Clamped stretch: straighten every bone along root->target. One pass,
        # no iteration - FABRIK cannot do better than this geometry.
        stretch_direction = (target_array - joints[0]) / root_target_distance
        stretched = [joints[0].copy()]
        for length in lengths:
            stretched.append(stretched[-1] + stretch_direction * length)
        residual = float(np.linalg.norm(stretched[-1] - target_array))
        return FabrikSolution(
            joint_positions=tuple(
                Point.from_prevalidated_array(array=joint) for joint in stretched
            ),
            iterations_used=0,
            final_residual=residual,
            converged=False,
        )

    iterations_used = 0
    final_residual = float("nan")
    for _ in range(max_iterations):
        joints = backward_pass(joints=joints)
        joints = forward_pass(joints=joints)
        iterations_used += 1
        final_residual = float(np.linalg.norm(joints[-1] - target_array))
        if final_residual <= tolerance_mm:
            return FabrikSolution(
                joint_positions=tuple(
                    Point.from_prevalidated_array(array=joint) for joint in joints
                ),
                iterations_used=iterations_used,
                final_residual=final_residual,
                converged=True,
            )

    raise ValueError(
        f"FABRIK: did not converge within {max_iterations} iterations - "
        f"residual {final_residual:.6f} against tolerance {tolerance_mm}. "
        "The target is likely inside a fold the initial pose cannot escape; "
        "adjust the seed or the tolerance."
    )
