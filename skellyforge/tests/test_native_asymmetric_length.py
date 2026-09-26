"""Check the shortening preference through the actual connected Ceres problem."""
import itertools
import numpy as np
import pytest
from skellyforge import _native


def arguments(target_length=100.):
    reference=100.
    local=np.array(list(itertools.product([-20.,20.],[-20.,20.],[-50.,50.])))
    observed=local.copy();observed[:,2]*=target_length/reference
    return dict(local=[local.tolist()],observed=[[observed.tolist()]]*3,
        times=[0.,.1,.2],parent_indices=[],parent_attachments=[],child_attachments=[],
        position_scale=10.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.,
        initial_quaternions=[[[1.,0.,0.,0.]]]*3,initial_roots=[[0.,0.,0.]]*3,
        axial_reference_lengths=[reference],length_prior_fraction=.5)


def test_explicit_symmetric_scales_reproduce_existing_default():
    args=arguments(120.)
    default=_native.fit_chain_sequence(**args)
    explicit=_native.fit_chain_sequence(**args,lengthening_prior_fraction=.5)
    np.testing.assert_allclose(default.lengths,explicit.lengths,atol=1e-10)
    assert default.costs[-1]==pytest.approx(explicit.costs[-1],abs=1e-12)


def test_extension_is_penalized_more_without_changing_compression_or_reference():
    for target in [80.,100.,120.]:
        args=arguments(target)
        symmetric=_native.fit_chain_sequence(**args)
        biased=_native.fit_chain_sequence(**args,lengthening_prior_fraction=.25)
        assert symmetric.converged and biased.converged
        if target<=100.:
            np.testing.assert_allclose(symmetric.lengths,biased.lengths,atol=1e-5)
        else:
            assert np.all(np.array(biased.lengths)<np.array(symmetric.lengths))
            assert np.all(np.array(biased.lengths)>100.)
        assert biased.parameter_blocks==symmetric.parameter_blocks
        assert biased.residual_blocks==symmetric.residual_blocks


@pytest.mark.parametrize('fraction',[0.,-1.,float('inf'),float('nan')])
def test_invalid_lengthening_scale_rejected(fraction):
    with pytest.raises(ValueError,match='finite and positive'):
        _native.fit_chain_sequence(**arguments(),lengthening_prior_fraction=fraction)


@pytest.mark.parametrize('lengths', [[10., 100., 300.], [300., 10., 100.]])
def test_free_lengths_recover_targets_outside_old_bounds_without_length_penalties(lengths):
    args=arguments()
    local=np.asarray(args['local'][0])
    args['observed']=[[(local * [1.,1.,length/100.]).tolist()] for length in lengths]
    constrained=_native.fit_chain_sequence(**args)
    free=_native.fit_chain_sequence(**args,free_axial_lengths=True)
    assert free.converged
    np.testing.assert_allclose(np.asarray(free.lengths)[:,0],lengths,atol=1e-5)
    assert free.landmark_cost < 1e-10
    assert free.length_prior_cost == 0.
    assert free.length_acceleration_cost == 0.
    assert constrained.residual_blocks-free.residual_blocks == 4
    assert constrained.parameter_blocks == free.parameter_blocks
    assert constrained.landmark_cost > free.landmark_cost
