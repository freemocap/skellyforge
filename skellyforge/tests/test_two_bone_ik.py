"""Two-bone IK: the analytic limb solver, verified against hand-built triangles."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain import solve_two_bone_ik

UPPER = 300.0
LOWER = 250.0


def _root() -> Point:
    return Point.from_xyz(x=100.0, y=200.0, z=300.0)


def _bone_lengths_hold(solution) -> None:
    upper = float(
        np.linalg.norm(
            solution.middle_joint.array - _root().array
        )
    )
    lower = float(
        np.linalg.norm(
            solution.end_joint.array - solution.middle_joint.array
        )
    )
    assert upper == pytest.approx(UPPER, abs=1e-9)
    assert lower == pytest.approx(LOWER, abs=1e-9)


def test_reachable_target_lands_the_end_exactly_on_it() -> None:
    target = Point.from_xyz(x=100.0 + 150.0, y=200.0, z=300.0 + 180.0)
    solution = solve_two_bone_ik(
        root_position=_root(),
        target=target,
        upper_length=UPPER,
        lower_length=LOWER,
    )
    np.testing.assert_allclose(solution.end_joint.array, target.array, atol=1e-9)
    _bone_lengths_hold(solution)


def test_elbow_bends_toward_the_pole_hint() -> None:
    """Same target, two opposite poles: the elbow lands on opposite sides."""
    target = Point.from_xyz(x=250.0, y=200.0, z=380.0)
    up = solve_two_bone_ik(
        root_position=_root(),
        target=target,
        upper_length=UPPER,
        lower_length=LOWER,
        pole_hint=np.array([0.0, 0.0, 1.0]),
    )
    down = solve_two_bone_ik(
        root_position=_root(),
        target=target,
        upper_length=UPPER,
        lower_length=LOWER,
        pole_hint=np.array([0.0, 0.0, -1.0]),
    )
    axis = target.array - _root().array
    axis = axis / np.linalg.norm(axis)
    up_side = float((up.middle_joint.array - _root().array) @ np.cross(axis, [0, -1, 0]))
    down_side = float((down.middle_joint.array - _root().array) @ np.cross(axis, [0, -1, 0]))
    assert up_side * down_side < 0.0
    # And both are valid limbs.
    _bone_lengths_hold(up)
    _bone_lengths_hold(down)


def test_full_extension_is_straight_and_exact() -> None:
    direction = np.array([0.6, 0.64, 0.48])
    direction = direction / np.linalg.norm(direction)
    reach = UPPER + LOWER
    target = Point.from_prevalidated_array(array=_root().array + direction * reach)
    solution = solve_two_bone_ik(
        root_position=_root(), target=target, upper_length=UPPER, lower_length=LOWER
    )
    np.testing.assert_allclose(solution.end_joint.array, target.array, atol=1e-9)
    np.testing.assert_allclose(
        solution.middle_joint.array, _root().array + direction * UPPER, atol=1e-9
    )


def test_target_farther_than_reach_raises_with_distances_named() -> None:
    direction = np.array([1.0, 0.0, 0.0])
    target = Point.from_prevalidated_array(
        array=_root().array + direction * (UPPER + LOWER + 10.0)
    )
    with pytest.raises(ValueError, match="exceeds"):
        solve_two_bone_ik(
            root_position=_root(), target=target, upper_length=UPPER, lower_length=LOWER
        )


def test_target_closer_than_the_fold_limit_raises() -> None:
    direction = np.array([0.0, 1.0, 0.0])
    target = Point.from_prevalidated_array(
        array=_root().array + direction * abs(UPPER - LOWER) / 2.0
    )
    with pytest.raises(ValueError, match="closer than"):
        solve_two_bone_ik(
            root_position=_root(), target=target, upper_length=UPPER, lower_length=LOWER
        )


def test_coincident_root_and_target_raises() -> None:
    with pytest.raises(ValueError, match="coincides with the root"):
        solve_two_bone_ik(
            root_position=_root(), target=_root(), upper_length=UPPER, lower_length=LOWER
        )


def test_non_positive_lengths_raise() -> None:
    with pytest.raises(ValueError, match="length must be"):
        solve_two_bone_ik(
            root_position=_root(),
            target=Point.from_xyz(x=1.0, y=2.0, z=3.0),
            upper_length=0.0,
            lower_length=LOWER,
        )


def test_pole_parallel_to_axis_falls_back_deterministically() -> None:
    """A pole along the root->target axis carries no side information; the
    solver falls back to its deterministic perpendicular instead of dividing
    into rounding error."""
    direction = np.array([1.0, 0.0, 0.0])
    target = Point.from_prevalidated_array(
        array=_root().array + direction * 0.7 * (UPPER + LOWER)
    )
    solution = solve_two_bone_ik(
        root_position=_root(),
        target=target,
        upper_length=UPPER,
        lower_length=LOWER,
        pole_hint=direction,
    )
    _bone_lengths_hold(solution)


def test_two_bone_solution_feeds_fabrik_to_instant_convergence() -> None:
    """Cross-solver contract: the analytic solution is already a fixed point of
    FABRIK - seeding the iterative solver with it converges on the first pass.
    """
    from skellyforge.core.skeleton.chain import solve_fabrik

    target = Point.from_xyz(x=260.0, y=210.0, z=360.0)
    solution = solve_two_bone_ik(
        root_position=_root(), target=target, upper_length=UPPER, lower_length=LOWER
    )
    chain = [_root(), solution.middle_joint, solution.end_joint]
    fabrik = solve_fabrik(chain_positions=chain, target=target)

    assert fabrik.converged
    assert fabrik.iterations_used == 1
    np.testing.assert_allclose(
        fabrik.joint_positions[-1].array, target.array, atol=1e-9
    )
