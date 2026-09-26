"""Exact attachment, rigid geometry and coordinate-frame invariance."""

import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native


def fixture(noise):
    local = np.array(
        list(itertools.product([-30.0, 30.0], [-30.0, 30.0], [-100.0, 100.0]))
    )
    attachments = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, -100.0]])
    rotations = Rotation.from_quat(
        [[1.0, 0.0, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0]], scalar_first=True
    )
    joint = np.array([30.0, 40.0, 220.0])
    truth = np.array(
        [r.apply(local - a) + joint for r, a in zip(rotations, attachments)]
    )
    return (
        local,
        attachments,
        truth,
        truth + np.random.default_rng(7).normal(0, noise, truth.shape),
    )


def solve(local, attachments, observed):
    return _native.fit_linked_rigid(
        local_a=local.tolist(),
        observed_a=observed[0].tolist(),
        attachment_a=attachments[0].tolist(),
        local_b=local.tolist(),
        observed_b=observed[1].tolist(),
        attachment_b=attachments[1].tolist(),
    )


@pytest.mark.parametrize("noise", [0.0, 15.0])
def test_exact_linkage_and_reported_objective(noise):
    local, attachments, truth, observed = fixture(noise)
    result = solve(local, attachments, observed)
    assert result.converged
    fitted = []
    for q, t, a in zip(result.quaternions, result.translations, attachments):
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)
        r = Rotation.from_quat(q, scalar_first=True)
        np.testing.assert_allclose(r.apply(a) + t, result.joint, atol=1e-10)
        points = r.apply(local) + t
        np.testing.assert_allclose(
            np.linalg.norm(points[:, None] - points, axis=-1),
            np.linalg.norm(local[:, None] - local, axis=-1),
            atol=1e-10,
        )
        fitted.append(points)
    assert result.costs[-1] == pytest.approx(
        0.5 * np.sum((np.array(fitted) - observed) ** 2), abs=1e-7
    )
    assert result.costs[-1] <= result.costs[0] + 1e-9
    if noise == 0:
        np.testing.assert_allclose(fitted, truth, atol=1e-6)


def test_world_transform_invariance():
    local, attachments, _, observed = fixture(5.0)
    first = solve(local, attachments, observed)
    rotation = Rotation.from_quat([0.8, 0.0, 0.6, 0.0], scalar_first=True)
    offset = np.array([-100.0, 70.0, 200.0])
    transformed = (
        rotation.apply(observed.reshape(-1, 3)).reshape(observed.shape) + offset
    )
    second = solve(local, attachments, transformed)
    assert second.converged
    np.testing.assert_allclose(
        second.joint, rotation.apply(first.joint) + offset, atol=1e-4
    )
    assert second.costs[-1] == pytest.approx(first.costs[-1], rel=1e-8)


def test_invalid_attachment_rejected():
    local, attachments, _, observed = fixture(0.0)
    attachments[0, 0] = np.nan
    with pytest.raises(ValueError, match="Attachments must be finite"):
        solve(local, attachments, observed)


def sparse_solve(local, attachments, observed, count):
    return _native.fit_linked_rigid(
        local_a=local.tolist(),
        observed_a=observed[0].tolist(),
        attachment_a=attachments[0].tolist(),
        local_b=local[:count].tolist(),
        observed_b=observed[1, :count].tolist(),
        attachment_b=attachments[1].tolist(),
    )


def test_attachment_resolves_two_landmark_child():
    local, attachments, truth, observed = fixture(0.0)
    fit = sparse_solve(local, attachments, observed, 2)
    assert fit.converged
    r = Rotation.from_quat(fit.quaternions[1], scalar_first=True)
    np.testing.assert_allclose(
        r.apply(local) + fit.translations[1], truth[1], atol=1e-5
    )


def test_one_child_landmark_leaves_roll_undetermined():
    local, attachments, _, observed = fixture(0.0)
    fit = sparse_solve(local, attachments, observed, 1)
    assert fit.converged
    r = Rotation.from_quat(fit.quaternions[1], scalar_first=True)
    predicted = r.apply(local) + fit.translations[1]
    axis = predicted[0] - fit.joint
    axis /= np.linalg.norm(axis)
    # A 90-degree rotation around the attachment-to-observed-landmark line
    # preserves every observed target but moves the withheld landmarks.
    alternative = Rotation.from_quat(
        np.r_[np.sqrt(0.5), axis * np.sqrt(0.5)], scalar_first=True
    )
    changed = alternative.apply(predicted - fit.joint) + fit.joint
    np.testing.assert_allclose(changed[0], observed[1, 0], atol=1e-5)
    assert np.linalg.norm(changed[1] - predicted[1]) > 100
    np.testing.assert_allclose(
        r.apply(attachments[1]) + fit.translations[1], fit.joint, atol=1e-10
    )


def test_sparse_independent_fit_requires_explicit_opt_in():
    local, _, _, observed = fixture(0.0)
    kwargs = dict(
        local=local[:2].tolist(),
        observed=observed[1, :2].tolist(),
        quaternion=[1.0, 0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="at least three"):
        _native.fit_rigid(**kwargs)
    result = _native.fit_rigid(**kwargs, allow_underconstrained=True)
    assert result.converged
    assert result.costs[-1] < 1e-10
