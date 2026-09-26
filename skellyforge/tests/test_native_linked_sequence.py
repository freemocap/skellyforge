"""Temporal residual blocks must recover compatible motion and report their costs."""

import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native


def inputs(noise=0.0):
    local = np.array(
        list(itertools.product([-30.0, 30.0], [-30.0, 30.0], [-100.0, 100.0]))
    )
    times = np.array([0.0, 0.05, 0.12, 0.2, 0.3, 0.42, 0.55, 0.7, 0.9])
    attachments = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, -100.0]])
    joints = np.array([30.0, 40.0, 230.0]) + times[:, None] * [30.0, -20.0, 10.0]
    truth = []
    for time, joint in zip(times, joints):
        angles = [time * 0.2, 0.7 + time * 0.4]
        rotations = [
            Rotation.from_quat(
                [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0], scalar_first=True
            )
            for a in angles
        ]
        truth.append(
            [r.apply(local - a) + joint for r, a in zip(rotations, attachments)]
        )
    truth = np.array(truth)
    observed = truth + np.random.default_rng(7).normal(0, noise, truth.shape)
    return local, times, attachments, joints, truth, observed


def solve(data, distance_unit=1.0, time_unit=1.0):
    local, times, attachments, _, _, observed = data
    return _native.fit_linked_sequence(
        local_a=(local * distance_unit).tolist(),
        observed_a=(observed[:, 0] * distance_unit).tolist(),
        attachment_a=(attachments[0] * distance_unit).tolist(),
        local_b=(local * distance_unit).tolist(),
        observed_b=[
            (frame[1] * distance_unit).tolist() if i not in [3, 4, 5] else []
            for i, frame in enumerate(observed)
        ],
        attachment_b=(attachments[1] * distance_unit).tolist(),
        times=(times * time_unit).tolist(),
        position_scale=10.0 * distance_unit,
        linear_acceleration_scale=3000.0 * distance_unit / time_unit**2,
        angular_acceleration_scale=20.0 / time_unit**2,
    )


def test_gap_recovery_for_constant_velocities_and_exact_linkage():
    data = inputs()
    local, times, attachments, joints, truth, _ = data
    result = solve(data)
    assert result.converged
    assert result.parameter_blocks == 3 * len(times)
    assert result.residual_blocks == 16 * len(times) - 3 * 8 + 3 * (len(times) - 2)
    for i in range(len(times)):
        for b in range(2):
            q = result.quaternions[i][b]
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)
            r = Rotation.from_quat(q, scalar_first=True)
            t = result.translations[i][b]
            np.testing.assert_allclose(
                r.apply(attachments[b]) + t, result.joints[i], atol=1e-10
            )
            np.testing.assert_allclose(r.apply(local) + t, truth[i, b], atol=1e-5)
    np.testing.assert_allclose(result.joints, joints, atol=1e-5)


def test_cost_accounting_and_unit_time_scaling():
    data = inputs(5.0)
    a = solve(data)
    b = solve(data, distance_unit=0.001, time_unit=2.0)
    assert a.converged and b.converged
    assert a.costs[-1] == pytest.approx(
        a.landmark_cost
        + a.joint_acceleration_cost
        + a.parent_acceleration_cost
        + a.child_acceleration_cost,
        rel=1e-8,
    )
    assert b.costs[-1] == pytest.approx(2 * a.costs[-1], rel=1e-6)
    np.testing.assert_allclose(
        a.translations, np.array(b.translations) * 1000, atol=1e-3
    )


def test_unbounded_child_gap_rejected():
    local, times, attachments, _, _, observed = inputs()
    child = observed[:, 1].tolist()
    child[0] = []
    with pytest.raises(ValueError, match="bounded"):
        _native.fit_linked_sequence(
            local_a=local.tolist(),
            observed_a=observed[:, 0].tolist(),
            attachment_a=attachments[0].tolist(),
            local_b=local.tolist(),
            observed_b=child,
            attachment_b=attachments[1].tolist(),
            times=times.tolist(),
            position_scale=10.0,
            linear_acceleration_scale=3000.0,
            angular_acceleration_scale=20.0,
        )


def test_hidden_observations_cannot_affect_fit():
    data = inputs(5.0)
    first = solve(data)
    modified = list(data)
    modified[-1] = data[-1].copy()
    modified[-1][3:6, 1] = 1e9
    second = solve(modified)
    np.testing.assert_array_equal(first.quaternions, second.quaternions)
    np.testing.assert_array_equal(first.joints, second.joints)
