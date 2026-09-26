"""Compiled quaternion fit checked against exact geometry and an SVD solution."""

import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native


def fixture():
    points = np.array(list(itertools.product([-100.0, 100.0], repeat=3)))
    q = np.array([0.5, 0.5, 0.5, 0.5])
    observed = Rotation.from_quat(q, scalar_first=True).apply(points) + [30, -70, 120]
    return points, observed


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_exact_fit_and_quaternion_sign(sign):
    points, observed = fixture()
    result = _native.fit_rigid(
        local=points.tolist(),
        observed=observed.tolist(),
        quaternion=[sign, 0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
    )
    fitted = (
        Rotation.from_quat(result.quaternion, scalar_first=True).apply(points)
        + result.translation
    )
    assert result.converged
    np.testing.assert_allclose(fitted, observed, atol=1e-4)
    assert np.linalg.norm(result.quaternion) == pytest.approx(1.0)


def test_noisy_partial_fit_matches_independent_svd():
    points, observed = fixture()
    points = points[:5]
    observed = observed[:5] + np.random.default_rng(42).normal(0, 5, (5, 3))
    result = _native.fit_rigid(
        local=points.tolist(),
        observed=observed.tolist(),
        quaternion=[1.0, 0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
    )
    u, _, vt = np.linalg.svd(
        (points - points.mean(0)).T @ (observed - observed.mean(0))
    )
    correction = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    reference = (points - points.mean(0)) @ rotation.T + observed.mean(0)
    fitted = (
        Rotation.from_quat(result.quaternion, scalar_first=True).apply(points)
        + result.translation
    )
    np.testing.assert_allclose(fitted, reference, atol=1e-3)
    assert result.costs[-1] == pytest.approx(
        0.5 * np.sum((fitted - observed) ** 2), rel=1e-8
    )


def test_reject_collinear_input():
    with pytest.raises(ValueError, match="non-collinear"):
        _native.fit_rigid(
            local=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            observed=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            translation=[0.0, 0.0, 0.0],
        )
