"""Time-aware Ceres fit: objective accounting, units/timing, and noisy data."""

import itertools
import numpy as np
import pytest
from skellyforge import _native


def data():
    local = np.array(list(itertools.product([-100.0, 100.0], repeat=3)))
    times = np.array([0.0, 0.04, 0.1, 0.17, 0.25, 0.4])
    observed = np.tile(local, (len(times), 1, 1)) + [30.0, 40.0, 180.0]
    observed += np.random.default_rng(7).normal(0, 5, observed.shape)
    return local, times, observed


def solve(local, times, observed, unit=1.0, time_unit=1.0, model="velocity"):
    order = 2 if model == "acceleration" else 1
    return _native.fit_rigid_sequence(
        local=(local * unit).tolist(),
        observed=(observed * unit).tolist(),
        times=(times * time_unit).tolist(),
        position_scale=10.0 * unit,
        linear_motion_scale=100.0 * unit / time_unit**order,
        angular_motion_scale=1.0 / time_unit**order,
        temporal_model=model,
    )


def test_static_sequence_reduces_error_and_reports_costs():
    local, times, observed = data()
    result = solve(local, times, observed)
    independent = [
        _native.fit_rigid(
            local=local.tolist(),
            observed=f.tolist(),
            quaternion=[1.0, 0.0, 0.0, 0.0],
            translation=[0.0, 0.0, 0.0],
        )
        for f in observed
    ]
    initial = np.array([r.translation for r in independent])
    fitted = np.array(result.translations)
    assert result.converged
    assert np.mean((fitted - [30, 40, 180]) ** 2) < np.mean(
        (initial - [30, 40, 180]) ** 2
    )
    np.testing.assert_allclose(
        np.linalg.norm(result.quaternions, axis=1), 1.0, atol=1e-12
    )
    assert result.costs[-1] == pytest.approx(
        result.landmark_cost + result.translation_cost + result.rotation_cost
    )


@pytest.mark.parametrize("model", ["velocity", "acceleration"])
def test_unit_and_time_scaling(model):
    local, times, observed = data()
    a = solve(local, times, observed, model=model)
    b = solve(local, times, observed, unit=0.001, time_unit=2.0, model=model)
    np.testing.assert_allclose(
        a.translations, np.array(b.translations) * 1000, atol=1e-3
    )
    assert b.costs[-1] == pytest.approx(a.costs[-1] * 2, rel=1e-6)


def test_actual_motion_is_not_frozen():
    local, times, observed = data()
    observed = np.tile(local, (len(times), 1, 1))
    observed[:, :, 0] += times[:, None] * 100
    result = solve(local, times, observed)
    assert result.converged
    assert np.ptp(np.array(result.translations)[:, 0]) > 30


def test_duplicate_timestamps_rejected():
    local, times, observed = data()
    times[1] = times[0]
    with pytest.raises(ValueError, match="increase strictly"):
        solve(local, times, observed)


def test_acceleration_preserves_constant_translation_and_angular_velocity():
    from scipy.spatial.transform import Rotation

    local, times, _ = data()
    # Unequal intervals, nonidentity starting pose, arbitrary world axis.
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    angles = times * 0.7
    increments = Rotation.from_quat(
        np.column_stack([np.cos(angles / 2), np.sin(angles / 2)[:, None] * axis]),
        scalar_first=True,
    )
    rotations = increments * Rotation.from_quat([0.8, 0.6, 0, 0], scalar_first=True)
    translations = np.array([30, 40, 180]) + times[:, None] * [100, -40, 20]
    observed = np.array([r.apply(local) + t for r, t in zip(rotations, translations)])
    result = solve(local, times, observed, model="acceleration")
    assert result.converged
    np.testing.assert_allclose(result.translations, translations, atol=1e-6)
    recovered = Rotation.from_quat(result.quaternions, scalar_first=True)
    np.testing.assert_allclose(recovered.as_matrix(), rotations.as_matrix(), atol=1e-8)
    assert result.translation_cost < 1e-12
    assert result.rotation_cost < 1e-12
