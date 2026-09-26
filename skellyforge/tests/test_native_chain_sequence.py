"""Chain FK, exact attachments and distal-to-upstream residual influence."""

import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native


def data(pulse=False, noise=0.0):
    local = np.array(
        list(itertools.product([-25.0, 25.0], [-25.0, 25.0], [-80.0, 80.0]))
    )
    times = np.array([0.0, 0.05, 0.12, 0.2, 0.3, 0.42, 0.55, 0.7, 0.9])
    parent = np.array([[10.0, 3.0, 80.0], [-5.0, 0.0, 60.0]])
    child = np.array([[-3.0, 1.0, -70.0], [1.0, 2.0, -50.0]])
    truth = []
    true_q = []
    true_t = []
    for i, time in enumerate(times):
        angles = [0.1 * time, 0.2 * time, 0.3 * time]
        if pulse:
            angles = [0.0, 0.4 if i == 4 else 0.0, 0.0]
        rotations = [
            Rotation.from_quat(
                [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0], scalar_first=True
            )
            for a in angles
        ]
        translations = [
            np.array([20.0, 30.0, 100.0]) + time * np.array([10.0, -20.0, 30.0])
        ]
        for b in range(1, 3):
            translations.append(
                translations[b - 1]
                + rotations[b - 1].apply(parent[b - 1])
                - rotations[b].apply(child[b - 1])
            )
        truth.append([r.apply(local) + t for r, t in zip(rotations, translations)])
        true_q.append([r.as_quat(scalar_first=True) for r in rotations])
        true_t.append(translations)
    truth = np.array(truth)
    observed = truth + np.random.default_rng(7).normal(0, noise, truth.shape)
    observed = [
        [
            body.tolist() if b != 1 or i not in [3, 4, 5] else []
            for b, body in enumerate(frame)
        ]
        for i, frame in enumerate(observed)
    ]
    return (
        local,
        times,
        parent,
        child,
        truth,
        observed,
        np.array(true_q),
        np.array(true_t),
    )


def solve(d, distance=1.0, time=1.0, motion_scale=20.0):
    local, times, parent, child, _, observed, _, _ = d
    return _native.fit_chain_sequence(
        local=[(local * distance).tolist()] * 3,
        observed=[
            [(np.array(points) * distance).tolist() for points in frame]
            for frame in observed
        ],
        parent_attachments=(parent * distance).tolist(),
        child_attachments=(child * distance).tolist(),
        times=(times * time).tolist(),
        position_scale=10.0 * distance,
        linear_acceleration_scale=3000.0 * distance / time**2,
        angular_acceleration_scale=motion_scale / time**2,
    )


def test_chain_pose_recovery_and_exact_attachments():
    d = data()
    local, times, parent, child, truth, _, _, _ = d
    r = solve(d)
    assert r.converged
    assert r.parameter_blocks == 4 * len(times)
    assert r.residual_blocks == 24 * len(times) - 24 + 4 * (len(times) - 2)
    for i in range(len(times)):
        rotations = [Rotation.from_quat(q, scalar_first=True) for q in r.quaternions[i]]
        np.testing.assert_allclose(
            np.linalg.norm(r.quaternions[i], axis=1), 1.0, atol=1e-12
        )
        for b in range(3):
            np.testing.assert_allclose(
                rotations[b].apply(local) + r.translations[i][b], truth[i, b], atol=1e-5
            )
        for b in range(2):
            np.testing.assert_allclose(
                rotations[b].apply(parent[b]) + r.translations[i][b],
                rotations[b + 1].apply(child[b]) + r.translations[i][b + 1],
                atol=1e-10,
            )


def test_distal_residual_moves_unobserved_middle_attachment_vector():
    d = data(pulse=True)
    parent, child = d[2:4]
    r = solve(d, motion_scale=1e5)
    assert r.converged
    local_vector = parent[1] - child[0]
    known = Rotation.from_quat(d[6][4, 1], scalar_first=True).apply(local_vector)
    initial = Rotation.from_quat(r.initial_quaternions[4][1], scalar_first=True).apply(
        local_vector
    )
    fitted = Rotation.from_quat(r.quaternions[4][1], scalar_first=True).apply(
        local_vector
    )
    assert np.linalg.norm(initial - known) > 40
    np.testing.assert_allclose(fitted, known, atol=0.02)


def test_costs_and_distance_time_scaling():
    d = data(noise=3.0)
    a = solve(d)
    b = solve(d, distance=0.001, time=2.0)
    assert a.converged and b.converged
    assert a.costs[-1] == pytest.approx(
        a.landmark_cost + a.root_acceleration_cost + sum(a.angular_acceleration_costs),
        rel=1e-8,
    )
    assert b.costs[-1] == pytest.approx(a.costs[-1] * 2, rel=1e-6)
    np.testing.assert_allclose(
        a.translations, np.array(b.translations) * 1000, atol=1e-3
    )


def test_duplicate_time_rejected():
    d = list(data())
    d[1] = d[1].copy()
    d[1][1] = d[1][0]
    with pytest.raises(ValueError, match="increase strictly"):
        solve(d)
