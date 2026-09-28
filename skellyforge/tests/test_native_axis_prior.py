import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.tests.test_native_length_proportion import inputs
from scripts.solver_window_sequence import fit_windows


def setup(angle=0., axis='z'):
    args,_=inputs()
    q=Rotation.from_euler(axis,angle,degrees=True).as_quat(scalar_first=True).tolist()
    args['initial_quaternions']=[[q]*3 for _ in range(3)]
    args['initial_roots']=[[0.,0.,0.]]*3
    p=_native.SegmentAxisPrior();p.segment=1;p.frames=[[1.,0.,0.]]*3;p.scale=.5
    return args,p


def test_cost_is_signed_axis_chord_not_bend_penalty():
    for angle in (0.,30.,-30.,120.,180.):
        args,p=setup(angle)
        options=_native.ChainSolveOptions();options.evaluate_only=True
        result=_native.fit_chain_sequence(**args,segment_axis_prior=p,solve_options=options)
        assert result.axis_prior_cost==pytest.approx((1-np.cos(np.radians(angle)))/p.scale**2,abs=1e-10)
    args,p=setup(50.,'x')
    result=_native.fit_chain_sequence(**args,segment_axis_prior=p,solve_options=options)
    assert result.axis_prior_cost<1e-12


def test_missing_evidence_and_window_slicing():
    args,p=setup()
    args['times']=[i*.5 for i in range(7)]
    for key in ('observed','initial_quaternions','initial_roots'):args[key]=[args[key][0]]*7
    p.frames=[[1.,0.,0.],None,[1.,0.,0.],None,[1.,0.,0.],None,[1.,0.,0.]]
    args['segment_axis_prior']=p
    result=fit_windows(args)
    assert np.isfinite(result.axis_prior_cost)
    assert len(result.lengths)==7


def test_invalid_basis_is_rejected():
    args,p=setup();p.local_anterior=[1.,0.,0.]
    with pytest.raises(ValueError,match='axis prior'):
        _native.fit_chain_sequence(**args,segment_axis_prior=p)
