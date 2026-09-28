import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.tests.test_native_length_proportion import inputs
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows


def prepared():
    args,_=inputs()
    result=_native.fit_chain_sequence(**args)
    args.update(initial_quaternions=result.quaternions,initial_roots=result.roots)
    options=_native.ChainSolveOptions()
    options.initial_lengths=result.lengths
    options.evaluate_only=True
    return args,result,options


@pytest.mark.parametrize('distance',[20.,0.,-20.])
def test_half_space_sign_cost_and_rigid_coordinate_invariance(distance):
    args,result,options=prepared()
    p=_native.LandmarkHalfSpacePrior();p.segment=2;p.local_point=[0,0,0];p.scale=20.
    normal=np.array([1.,0.,0.])
    p.frames=[[np.asarray(t[2])-distance*normal,normal] for t in result.translations]
    fit=_native.fit_chain_sequence(**args,solve_options=options,landmark_half_space_prior=p)
    assert fit.half_space_cost==pytest.approx(.5*(max(-distance,0)/20.)**2,abs=1e-10)
    rotation=Rotation.from_euler('xyz',[30,40,50],degrees=True);shift=np.array([90,80,70])
    args['observed']=[[(rotation.apply(body)+shift).tolist() for body in frame] for frame in args['observed']]
    args['initial_roots']=(rotation.apply(args['initial_roots'])+shift).tolist()
    args['initial_quaternions']=[(rotation*Rotation.from_quat(frame,scalar_first=True)).as_quat(scalar_first=True).tolist() for frame in args['initial_quaternions']]
    p.frames=[[(rotation.apply(origin)+shift).tolist(),rotation.apply(axis).tolist()] for origin,axis in p.frames]
    moved=_native.fit_chain_sequence(**args,solve_options=options,landmark_half_space_prior=p)
    assert moved.half_space_cost==pytest.approx(fit.half_space_cost,abs=1e-10)


def test_half_space_missing_frame_validation_and_window_slicing():
    args,result,options=prepared()
    p=_native.LandmarkHalfSpacePrior();p.segment=2;p.scale=20.
    p.frames=[[[20.,0,0],[1.,0,0]],None,[[20.,0,0],[1.,0,0]]]
    fit=_native.fit_chain_sequence(**args,solve_options=options,landmark_half_space_prior=p)
    assert fit.half_space_cost==pytest.approx(.25,abs=1e-8)
    # Five frames exercise two fixed history states, slicing and final evaluation.
    for key in ('observed','initial_roots','initial_quaternions'):args[key]+=args[key][-2:]
    args['times']=[0,.5,1.,1.5,2.]
    p.frames=p.frames+[None,p.frames[-1]]
    window=fit_windows(dict(args,landmark_half_space_prior=p))
    assert window.usable and len(window.processing['windows'])==3
    assert np.isfinite(window.half_space_cost)
    p.frames=[[[0,0,0],[2.,0,0]]]*5
    with pytest.raises(ValueError,match='unit length'):
        _native.fit_chain_sequence(**args,landmark_half_space_prior=p)


def test_optimization_reduces_posterior_violation_without_attracting_anterior_points():
    args,result,_=prepared()
    p=_native.LandmarkHalfSpacePrior();p.segment=0;p.scale=.1
    p.frames=[[[20.,0,0],[1.,0,0]]]*3
    penalized=_native.fit_chain_sequence(**args,landmark_half_space_prior=p)
    before=np.maximum(20-np.asarray(result.roots)[:,0],0)
    after=np.maximum(20-np.asarray(penalized.roots)[:,0],0)
    assert np.linalg.norm(after)<.1*np.linalg.norm(before)
    p.frames=[[[-20.,0,0],[1.,0,0]]]*3
    anterior=_native.fit_chain_sequence(**args,landmark_half_space_prior=p)
    assert anterior.half_space_cost==0
    np.testing.assert_allclose(anterior.roots,result.roots,atol=1e-4)
