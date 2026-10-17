"""FABRIK: convergence, length preservation, and loud failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.fabrik_ik import solve_fabrik


def _straight_chain(*, lengths: list[float], direction=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 100.0)):
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    origin = np.asarray(origin, dtype=np.float64)
    points = [origin]
    for length in lengths:
        points.append(points[-1] + unit * length)
    return [Point.from_prevalidated_array(array=point) for point in points]


def _bone_lengths_hold(joints, lengths: list[float]) -> None:
    for index, length in enumerate(lengths):
        actual = float(np.linalg.norm(joints[index + 1].array - joints[index].array))
        assert actual == pytest.approx(length, abs=1e-9)


def test_reachable_target_converges_exactly() -> None:
    lengths = [200.0, 180.0, 120.0]
    chain = _straight_chain(lengths=lengths)
    total = sum(lengths)
    target = Point.from_xyz(x=60.0, y=40.0, z=total * 0.8)
    solution = solve_fabrik(chain_positions=chain, target=target)

    assert solution.converged
    assert solution.final_residual <= 0.01
    assert solution.iterations_used >= 1
    _bone_lengths_hold(solution.joint_positions, lengths)
    # The root is pinned.
    np.testing.assert_allclose(solution.joint_positions[0].array, chain[0].array)


def test_target_on_the_end_already_needs_one_pass() -> None:
    lengths = [150.0, 150.0]
    chain = _straight_chain(lengths=lengths)
    target = Point.from_xyz(
        x=float(chain[-1].array[0]), y=float(chain[-1].array[1]), z=float(chain[-1].array[2])
    )
    solution = solve_fabrik(chain_positions=chain, target=target)
    assert solution.converged
    assert solution.iterations_used == 1
    np.testing.assert_allclose(
        solution.joint_positions[-1].array, target.array, atol=1e-9
    )


def test_out_of_reach_raises_by_default() -> None:
    lengths = [200.0, 200.0]
    chain = _straight_chain(lengths=lengths)
    far = Point.from_xyz(x=900.0, y=0.0, z=100.0)  # 800 mm away from the root at x=0
    with pytest.raises(ValueError, match="unreachable"):
        solve_fabrik(chain_positions=chain, target=far)


def test_out_of_reach_with_permission_straightens_and_reports() -> None:
    lengths = [200.0, 200.0]
    chain = _straight_chain(lengths=lengths)
    far = Point.from_xyz(x=900.0, y=0.0, z=100.0)
    solution = solve_fabrik(chain_positions=chain, target=far, allow_out_of_reach=True)

    assert not solution.converged
    assert solution.final_residual == pytest.approx(900.0 - sum(lengths), abs=1e-9)
    _bone_lengths_hold(solution.joint_positions, lengths)
    # Fully stretched along root->target.
    stretch_direction = (far.array - chain[0].array) / np.linalg.norm(
        far.array - chain[0].array
    )
    np.testing.assert_allclose(
        solution.joint_positions[-1].array,
        chain[0].array + stretch_direction * sum(lengths),
        atol=1e-9,
    )


def test_iteration_exhaustion_raises_rather_than_near_misses() -> None:
    lengths = [200.0, 200.0]
    chain = _straight_chain(lengths=lengths)
    target = Point.from_xyz(x=50.0, y=30.0, z=350.0)
    with pytest.raises(ValueError, match="did not converge"):
        solve_fabrik(
            chain_positions=chain,
            target=target,
            tolerance_mm=1e-12,
            max_iterations=2,
        )


def test_bent_seed_converges_to_a_folded_target() -> None:
    """A folded target behind the root: reachable, and FABRIK must find it."""
    lengths = [250.0, 250.0]
    chain = [
        Point.from_xyz(x=0.0, y=0.0, z=100.0),
        Point.from_xyz(x=250.0, y=0.0, z=100.0),
        Point.from_xyz(x=500.0, y=0.0, z=100.0),
    ]
    target = Point.from_xyz(x=-80.0, y=20.0, z=260.0)
    solution = solve_fabrik(chain_positions=chain, target=target)
    assert solution.converged
    assert solution.final_residual <= 0.01
    _bone_lengths_hold(solution.joint_positions, lengths)
    # The end genuinely reached the target's neighborhood.
    distance_to_target = float(
        np.linalg.norm(solution.joint_positions[-1].array - target.array)
    )
    assert distance_to_target <= 0.01


def test_degenerate_bone_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        solve_fabrik(
            chain_positions=[
                Point.from_xyz(x=0.0, y=0.0, z=0.0),
                Point.from_xyz(x=10.0, y=0.0, z=0.0),
                Point.from_xyz(x=10.0, y=0.0, z=0.0),
            ],
            target=Point.from_xyz(x=1.0, y=2.0, z=3.0),
        )


def test_single_joint_chains_are_rejected() -> None:
    with pytest.raises(ValueError, match="two joints"):
        solve_fabrik(
            chain_positions=[Point.from_xyz(x=0.0, y=0.0, z=0.0)],
            target=Point.from_xyz(x=1.0, y=1.0, z=1.0),
        )


def test_non_positive_tolerances_and_iteration_caps_raise() -> None:
    chain = _straight_chain(lengths=[100.0, 100.0])
    target = Point.from_xyz(x=50.0, y=0.0, z=150.0)
    with pytest.raises(ValueError, match="tolerance"):
        solve_fabrik(chain_positions=chain, target=target, tolerance_mm=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        solve_fabrik(chain_positions=chain, target=target, max_iterations=0)
