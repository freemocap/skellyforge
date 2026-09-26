"""Three axial lengths retain free total size with a soft proportion residual."""
import numpy as np
import pytest
from skellyforge import _native


def inputs():
    local=np.array([[0,0,0],[10,0,0],[0,10,0],[0,0,80.]])
    lengths=np.array([[180,200,63],[198,220,69.3],[216,240,75.6]])
    observed=[]
    for row in lengths:
        bodies=[];z=0.
        for length in row:
            points=local.copy();points[:,2]*=length/80.;points[:,2]+=z
            bodies.append(points.tolist());z+=length
        observed.append(bodies)
    return dict(local=[local.tolist()]*3,observed=observed,parent_attachments=[[0,0,80]]*2,
                child_attachments=[[0,0,0]]*2,parent_indices=[0,1],times=[0.,.5,1.],
                position_scale=1.,linear_acceleration_scale=1e6,angular_acceleration_scale=1e6,
                axial_reference_lengths=[80.]*3,free_axial_lengths=True),lengths


def prior(ratios=(18.,20.,6.3),segments=(0,1,2),scale=50.):
    p=_native.LengthProportionPrior();p.ratios=ratios;p.segments=segments;p.scale=scale
    return p


def test_recovers_variable_total_length_and_reports_exact_prior_cost():
    args,truth=inputs();p=prior()
    baseline=_native.fit_chain_sequence(**args)
    result=_native.fit_chain_sequence(**args,length_proportion_prior=p)
    assert result.converged
    np.testing.assert_allclose(result.lengths,truth,atol=1e-3)
    assert result.parameter_blocks==baseline.parameter_blocks
    assert result.residual_blocks==baseline.residual_blocks+3
    lengths=np.array(result.lengths);fractions=np.array(p.ratios)/sum(p.ratios)
    residual=(lengths-lengths.sum(axis=1)[:,None]*fractions)/p.scale
    assert result.length_proportion_cost==pytest.approx(.5*np.sum(np.array([.25,.5,.25])[:,None]*residual**2),abs=1e-10)
    assert result.length_equality_cost==result.length_prior_cost==result.length_acceleration_cost==0
    assert np.ptp(lengths.sum(axis=1))>80
    # Ratios are scale-free inputs, not millimeters or stature constraints.
    rescaled=_native.fit_chain_sequence(**args,length_proportion_prior=prior((180.,200.,63.)))
    np.testing.assert_allclose(result.lengths,rescaled.lengths,atol=1e-5)


def test_proportion_strength_is_soft_and_not_a_new_length_measurement():
    args,_=inputs()
    for frame in args['observed']:frame[2][-1][2]+=30.
    weak=_native.fit_chain_sequence(**args,length_proportion_prior=prior(scale=50.))
    strong=_native.fit_chain_sequence(**args,length_proportion_prior=prior(scale=.01))
    fractions=np.array([18.,20.,6.3])/44.3
    def deviation(result):
        lengths=np.array(result.lengths)
        return np.sqrt(np.mean((lengths-lengths.sum(1)[:,None]*fractions)**2))
    assert deviation(weak)>1.
    assert deviation(strong)<.01
    assert strong.landmark_cost>weak.landmark_cost
    for i in range(3):
        for child,parent in enumerate((0,1),1):
            from scipy.spatial.transform import Rotation
            endpoint=np.array(strong.translations[i][parent])+Rotation.from_quat(strong.quaternions[i][parent],scalar_first=True).apply([0,0,strong.lengths[i][parent]])
            np.testing.assert_allclose(endpoint,strong.translations[i][child],atol=1e-9)


@pytest.mark.parametrize('ratios,segments,scale',[
    ((0,20,6.3),(0,1,2),50),((18,float('nan'),6.3),(0,1,2),50),
    ((18,20,6.3),(0,0,2),50),((18,20,6.3),(-1,1,2),50),
    ((18,20,6.3),(0,1,3),50),((18,20,6.3),(0,1,2),0),
])
def test_invalid_prior_rejected(ratios,segments,scale):
    args,_=inputs()
    with pytest.raises(ValueError,match='Length proportion'):
        _native.fit_chain_sequence(**args,length_proportion_prior=prior(ratios,segments,scale))
