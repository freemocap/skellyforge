"""Equal-length preference couples two scalar blocks, not their sum or time."""
import numpy as np
import pytest
from skellyforge import _native
from skellyforge.tests.test_native_axial import fixture


def prior(a=1,b=2,scale=50.):
    value=_native.LengthEqualityPrior();value.segment_a=a;value.segment_b=b;value.scale=scale
    return value


def test_soft_equality_reduces_difference_without_fixing_total_length():
    args,_,refs,_=fixture();args['free_axial_lengths']=True
    baseline=_native.fit_chain_sequence(**args)
    result=_native.fit_chain_sequence(**args,length_equality_prior=prior(scale=.01))
    assert baseline.converged and result.converged
    a=np.asarray(baseline.lengths)[:,1:3];b=np.asarray(result.lengths)[:,1:3]
    assert np.mean(abs(a[:,0]-a[:,1]))>7
    assert np.max(abs(b[:,0]-b[:,1]))<.01
    # Equality does not reinstate the sum of the reference lengths (160 mm).
    assert np.max(b.sum(axis=1))<140
    assert result.parameter_blocks==baseline.parameter_blocks
    assert result.residual_blocks==baseline.residual_blocks+len(args['times'])
    dt=np.diff(args['times']);weights=np.r_[dt,0]/2+np.r_[0,dt]/2
    expected=.5*np.sum(weights*((b[:,0]-b[:,1])/.01)**2)
    assert result.length_equality_cost==pytest.approx(expected,rel=1e-8,abs=1e-12)
    assert baseline.length_equality_cost==0
    assert result.length_prior_cost==0 and result.length_acceleration_cost==0


def test_weak_equality_remains_soft_and_is_symmetric():
    args,*_=fixture();args['free_axial_lengths']=True
    a=_native.fit_chain_sequence(**args,length_equality_prior=prior())
    b=_native.fit_chain_sequence(**args,length_equality_prior=prior(a=2,b=1))
    np.testing.assert_allclose(a.lengths,b.lengths,atol=1e-6)
    assert np.mean(abs(np.asarray(a.lengths)[:,1]-np.asarray(a.lengths)[:,2]))>1


@pytest.mark.parametrize('a,b,scale',[(1,1,50),(0,1,50),(-1,1,50),(1,5,50),(1,2,0),(1,2,float('nan'))])
def test_invalid_equality_definition_is_rejected(a,b,scale):
    args,*_=fixture()
    with pytest.raises(ValueError,match='Length equality'):
        _native.fit_chain_sequence(**args,length_equality_prior=prior(a,b,scale))
