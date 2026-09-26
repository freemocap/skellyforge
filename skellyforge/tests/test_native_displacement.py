"""Bounded scalar linkage displacement must not deform rigid segment geometry."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.tests.test_native_chain_sequence import data


def solve(extension=12.0, enabled=True, bound=40.0, distance=1.0, time=1.0):
    local, times, parent, child, truth, _, quaternions, _ = data()
    observed = truth.copy()
    for i in range(len(times)):
        observed[i, 2] += Rotation.from_quat(
            quaternions[i, 1], scalar_first=True
        ).apply([0.0, 0.0, extension])
    result = _native.fit_chain_sequence(
        local=[(local * distance).tolist()] * 3,
        observed=(observed * distance).tolist(),
        parent_attachments=(parent * distance).tolist(),
        child_attachments=(child * distance).tolist(),
        times=(times * time).tolist(),
        position_scale=10.0 * distance,
        linear_acceleration_scale=3000.0 * distance / time**2,
        angular_acceleration_scale=20.0 / time**2,
        allow_displacement=enabled,
        displacement_scale=1e6 * distance,
        displacement_acceleration_scale=500.0 * distance / time**2,
        displacement_bound=bound * distance,
    )
    return result, local, parent, child, observed


def test_displacement_recovery_and_rigid_segment_geometry():
    result, local, parent, child, observed = solve()
    assert result.converged
    np.testing.assert_allclose(result.displacements, 12.0, atol=1e-4)
    for i in range(len(observed)):
        rotations = [
            Rotation.from_quat(q, scalar_first=True) for q in result.quaternions[i]
        ]
        for b in range(3):
            predicted = rotations[b].apply(local) + result.translations[i][b]
            np.testing.assert_allclose(predicted, observed[i, b], atol=1e-4)
            np.testing.assert_allclose(
                np.linalg.norm(predicted[:, None] - predicted, axis=-1),
                np.linalg.norm(local[:, None] - local, axis=-1),
                atol=1e-10,
            )
        gap = (
            rotations[2].apply(child[1])
            + result.translations[i][2]
            - rotations[1].apply(parent[1])
            - result.translations[i][1]
        )
        np.testing.assert_allclose(
            gap, rotations[1].apply([0, 0, result.displacements[i]]), atol=1e-10
        )
    assert result.parameter_blocks == 5 * len(observed)
    assert result.residual_blocks == 24 * len(observed) + 4 * (len(observed) - 2) + len(
        observed
    ) + (len(observed) - 2)


def test_zero_displacement_matches_fixed_linkage():
    fixed, *_ = solve(extension=0.0, enabled=False)
    relaxed, *_ = solve(extension=0.0)
    np.testing.assert_allclose(relaxed.displacements, 0.0, atol=1e-6)
    np.testing.assert_allclose(relaxed.translations, fixed.translations, atol=1e-5)


def test_parameter_bound_is_enforced_and_cost_accounting_matches():
    result, *_ = solve(extension=60.0, bound=10.0)
    assert result.converged
    assert max(abs(d) for d in result.displacements) <= 10.0 + 1e-10
    assert max(result.displacements) == pytest.approx(10.0, abs=1e-6)
    assert result.costs[-1] == pytest.approx(
        result.landmark_cost
        + result.root_acceleration_cost
        + sum(result.angular_acceleration_costs)
        + result.displacement_prior_cost
        + result.displacement_acceleration_cost,
        rel=1e-8,
    )


def test_displacement_distance_and_time_units():
    a, *_ = solve(extension=60.0, bound=10.0)
    b, *_ = solve(extension=60.0, bound=10.0, distance=0.001, time=2.0)
    assert b.converged
    np.testing.assert_allclose(
        a.displacements, np.array(b.displacements) * 1000, atol=1e-4
    )
    np.testing.assert_allclose(
        a.translations, np.array(b.translations) * 1000, atol=1e-3
    )
    assert b.costs[-1] == pytest.approx(2 * a.costs[-1], rel=1e-6)


def test_missing_middle_roll_is_unobserved_but_axial_displacement_is_determined():
    """Actual axial attachment geometry has a roll ambiguity, not a length ambiguity."""
    from scripts.solver_chain_experiment import chain_inputs

    local, parent, child, _, _, records = chain_inputs(noise=0, gap=5, extension=20)
    record = records[20]
    rotations = record["rotations"]
    translations = np.array(record["translations"])
    displacement = record["displacement"]
    a = rotations[0].apply(parent[0]) + translations[0]
    b = rotations[2].apply(child[1]) + translations[2]
    assert np.linalg.norm(b - a) - 160 == pytest.approx(displacement, abs=1e-10)
    # Right multiplication changes local-Z roll while preserving the attachment axis.
    roll = Rotation.from_quat([np.cos(0.6), 0, 0, np.sin(0.6)], scalar_first=True)
    alternative = rotations[1] * roll
    middle_translation = a - alternative.apply(child[0])
    distal_translation = (
        middle_translation + alternative.apply(parent[1] + [0, 0, displacement])
        - rotations[2].apply(child[1])
    )
    np.testing.assert_allclose(distal_translation, translations[2], atol=1e-10)
    assert np.linalg.norm(
        alternative.apply(local) + middle_translation
        - (rotations[1].apply(local) + translations[1])
    ) > 10


def test_displacement_and_gap_use_identical_observations_in_both_problems():
    from scripts.solver_displacement_experiment import displacement_experiment

    run = displacement_experiment(noise=1, extension=20, gap=5)
    fixed = run["methods"]["fixed"]
    fitted = run["methods"]["displacement"]
    assert fitted["summary"]["Ceres parameter blocks"] == 205
    assert fitted["summary"]["Ceres residual blocks"] == 1180
    assert fixed["summary"]["Ceres residual blocks"] == 1100
    for i, (a, b) in enumerate(zip(fixed["frames"], fitted["frames"])):
        assert a["converged"] and b["converged"]
        assert [x["observed"] for x in a["bodies"]] == [x["observed"] for x in b["bodies"]]
        assert b["diagnostics"]["Second linkage equation error (mm)"] < 1e-9
        assert abs(b["displacement"]) <= 40
        if i in run["gap_frames"]:
            assert b["bodies"][1]["observed"] == [None] * 8
            assert b["diagnostics"]["Middle landmark residual blocks"] == 0
    assert fitted["summary"]["All landmark RMS vs known (mm)"] < fixed["summary"]["All landmark RMS vs known (mm)"]
