"""Line preferences: analytic translation optimum, side sign, and frame invariance."""
import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts.solver_chest_line import mapped_centerline, LINE_SOURCE_LANDMARKS


def solve(front, rotation=None, offset=None, enabled=True):
    rotation = Rotation.identity() if rotation is None else rotation
    offset = np.zeros(3) if offset is None else offset
    local=np.array(list(itertools.product([-10.,10.],repeat=3)))
    target=np.array([20.,front,30.])
    args=dict(local=[local.tolist()],observed=[[ (rotation.apply(local+target)+offset).tolist() ]]*3,
        times=[0.,.1,.2],parent_indices=[],parent_attachments=[],child_attachments=[],
        position_scale=10.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.,
        initial_quaternions=[[rotation.as_quat(scalar_first=True).tolist()]]*3,
        initial_roots=[offset.tolist()]*3)
    p=_native.LandmarkLinePrior();p.segment=0;p.local_point=[0.,0.,0.]
    p.frames=[[offset.tolist(),rotation.apply([1.,0.,0.]).tolist(),rotation.apply([0.,1.,0.]).tolist()]]*3
    p.distance_scale=10.;p.anterior_scale=5.
    return _native.fit_chain_sequence(**args,landmark_line_prior=p if enabled else None),args,p


@pytest.mark.parametrize('front',[-20.,0.,20.])
def test_line_prior_has_analytic_optimum_and_does_not_pin_height(front):
    fit,_,_=solve(front)
    expected=[20.*.08/.09,front*.08/(.09+(.04 if front>0 else 0)),30.]
    assert fit.converged
    np.testing.assert_allclose(fit.roots,[expected]*3,atol=1e-5)
    baseline,_,_=solve(front,enabled=False)
    assert fit.parameter_blocks==baseline.parameter_blocks
    assert fit.residual_blocks==baseline.residual_blocks+3
    assert fit.line_prior_cost>=0 and baseline.line_prior_cost==0


def test_line_prior_follows_rotating_translating_person():
    original,_,_=solve(20.)
    rotation=Rotation.from_quat([.3,.4,.1,.7]);offset=np.array([100.,-200.,300.])
    transformed,_,_=solve(20.,rotation,offset)
    np.testing.assert_allclose(transformed.roots,rotation.apply(original.roots)+offset,atol=1e-5)
    assert transformed.line_prior_cost==pytest.approx(original.line_prior_cost,abs=1e-8)


def test_line_prior_omits_only_unavailable_frames_and_rejects_invalid_basis():
    _,args,p=solve(20.)
    p.frames=[p.frames[0],None,p.frames[2]]
    fit=_native.fit_chain_sequence(**args,landmark_line_prior=p)
    baseline=_native.fit_chain_sequence(**args)
    assert fit.residual_blocks==baseline.residual_blocks+2
    p.frames=[[[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]]]*3
    with pytest.raises(ValueError,match='orthonormal'):
        _native.fit_chain_sequence(**args,landmark_line_prior=p)


def test_mapped_line_sign_and_degeneracy():
    sources=dict(zip(LINE_SOURCE_LANDMARKS,['lh','rh','ls','rs']))
    points=dict(lh=[-10,0,0],rh=[10,0,0],ls=[-20,0,100],rs=[20,0,100])
    frame=mapped_centerline(points,sources)
    np.testing.assert_allclose(frame['anterior'],[0,1,0])
    np.testing.assert_allclose(frame['lateral'],[1,0,0])
    rotation=Rotation.from_quat([.3,.4,.1,.7]);offset=np.array([100.,-200.,300.])
    moved=mapped_centerline({k:rotation.apply(v)+offset for k,v in points.items()},sources)
    for name in ['up','lateral','anterior']:
        np.testing.assert_allclose(moved[name],rotation.apply(frame[name]),atol=1e-12)
    assert mapped_centerline({k:v for k,v in points.items() if k!='ls'},sources) is None
    points['rh']=points['lh']
    assert mapped_centerline(points,sources) is None
