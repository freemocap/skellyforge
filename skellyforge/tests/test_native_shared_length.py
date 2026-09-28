"""Exact length parameterization, FK consistency and fixed-window history."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.tests.test_native_length_proportion import inputs
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows


def shared():
    group = _native.SharedAxialLength()
    group.segments = [0, 1, 2]
    group.ratios = [18., 20., 6.3]
    return group


def test_shared_parameter_recovers_truth_and_preserves_fk():
    args, truth = inputs()
    independent = _native.fit_chain_sequence(**args)
    result = _native.fit_chain_sequence(**args, shared_axial_length=shared())
    assert result.converged
    assert result.parameter_blocks == independent.parameter_blocks - 2*len(truth)
    assert result.residual_blocks == independent.residual_blocks
    np.testing.assert_allclose(result.lengths, truth, atol=1e-3)
    lengths = np.asarray(result.lengths)
    np.testing.assert_allclose(lengths/lengths.sum(1)[:, None], np.tile(np.array(shared().ratios)/44.3, (3, 1)), atol=1e-14)
    for i in range(3):
        for b in (0, 1):
            endpoint = np.asarray(result.translations[i][b]) + Rotation.from_quat(result.quaternions[i][b], scalar_first=True).apply([0, 0, lengths[i, b]])
            np.testing.assert_allclose(endpoint, result.translations[i][b+1], atol=1e-9)


def test_rest_cost_and_window_boundary_states():
    args, _ = inputs()
    args['times'] = [i*.5 for i in range(7)]
    args['observed'] = [args['observed'][0]]*7
    args['initial_quaternions'] = [[[1., 0., 0., 0.]]*3 for _ in args['times']]
    args['initial_roots'] = [[0., 0., 0.]]*7
    args.update(shared_axial_length=shared(), free_length_rest_prior=True, length_prior_fraction=.5)
    result = fit_windows(args)
    # fit_windows checks fixed history bit-for-bit and evaluates the assembled state.
    lengths = np.asarray(result.lengths)
    expected = .5*np.sum(((lengths-80)/40)**2*np.array([.25]+[.5]*5+[.25])[:, None])
    assert result.length_prior_cost == pytest.approx(expected)
    assert result.length_proportion_cost == result.length_acceleration_cost == 0
    np.testing.assert_allclose(lengths[:, 0]/18, lengths[:, 1]/20, atol=1e-12)


def test_evaluation_rejects_inconsistent_lengths():
    args, _ = inputs()
    options = _native.ChainSolveOptions()
    options.evaluate_only = True
    with pytest.raises(ValueError, match='must satisfy their ratios'):
        _native.fit_chain_sequence(**args, shared_axial_length=shared(), solve_options=options)


@pytest.mark.parametrize('segments,ratios', [([0, 0, 2], [1, 1, 1]), ([0, 1, 9], [1, 1, 1]), ([0, 1, 2], [1, 0, 1])])
def test_invalid_group(segments, ratios):
    group = shared(); group.segments = segments; group.ratios = ratios
    args, _ = inputs()
    with pytest.raises(ValueError, match='Shared axial length'):
        _native.fit_chain_sequence(**args, shared_axial_length=group)
