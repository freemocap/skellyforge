import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.tests.test_native_length_proportion import inputs
from skellyforge.tests.test_native_shared_length import shared


def test_shared_total_bounds_and_evaluation():
    args,_=inputs();group=shared();group.minimum_total=200.;group.maximum_total=250.
    result=_native.fit_chain_sequence(**args,shared_axial_length=group)
    assert result.usable
    totals=np.sum(result.lengths,axis=1)
    assert np.all(totals>=200.-1e-8) and np.all(totals<=250.+1e-8)
    assert np.max(totals)==pytest.approx(250.,abs=1e-6)
    options=_native.ChainSolveOptions();options.evaluate_only=True
    options.initial_lengths=(np.asarray(result.lengths)*2).tolist()
    with pytest.raises(ValueError,match='violates bounds'):
        _native.fit_chain_sequence(**args,shared_axial_length=group,solve_options=options)


@pytest.mark.parametrize('axis,angle,expected', [('z',60.,.5),('x',60.,0.),('y',100.,0.)])
def test_twist_cost_separates_swing_and_is_quaternion_sign_invariant(axis,angle,expected):
    args,_=inputs()
    q=Rotation.from_euler(axis,angle,degrees=True).as_quat(scalar_first=True)
    prior=_native.RelativeTwistPrior();prior.parent=0;prior.child=1;prior.scale=1.
    options=_native.ChainSolveOptions();options.evaluate_only=True
    for sign in (1,-1):
        args['initial_quaternions']=[[[1.,0.,0.,0.],(sign*q).tolist(),[1.,0.,0.,0.]]]*3
        args['initial_roots']=[[0.,0.,0.]]*3
        result=_native.fit_chain_sequence(**args,relative_twist_priors=[prior],solve_options=options)
        assert result.twist_prior_cost==pytest.approx(expected,abs=1e-10)


def test_invalid_bounds_and_twist_definition():
    args,_=inputs();group=shared();group.minimum_total=300.;group.maximum_total=200.
    with pytest.raises(ValueError,match='minimum exceeds'):
        _native.fit_chain_sequence(**args,shared_axial_length=group)
    p=_native.RelativeTwistPrior();p.parent=0;p.child=1;p.axis=[0.,0.,2.]
    with pytest.raises(ValueError,match='twist prior'):
        _native.fit_chain_sequence(**args,relative_twist_priors=[p])


def test_twist_uses_authored_rest_and_is_world_rotation_invariant():
    args,_=inputs()
    rest=Rotation.from_euler('x',25.,degrees=True)
    error=Rotation.from_euler('z',60.,degrees=True)
    p=_native.RelativeTwistPrior();p.parent=0;p.child=1;p.reference=rest.as_quat(scalar_first=True).tolist()
    options=_native.ChainSolveOptions();options.evaluate_only=True
    args['initial_roots']=[[0.,0.,0.]]*3
    for parent in (Rotation.identity(),Rotation.from_euler('xyz',[35.,-20.,80.],degrees=True)):
        args['initial_quaternions']=[[parent.as_quat(scalar_first=True).tolist(),
            (parent*rest*error).as_quat(scalar_first=True).tolist(),[1.,0.,0.,0.]]]*3
        result=_native.fit_chain_sequence(**args,relative_twist_priors=[p],solve_options=options)
        assert result.twist_prior_cost==pytest.approx(.5,abs=1e-10)


def test_bounded_twist_windows_preserve_committed_history():
    from scripts.solver_window_sequence import fit_windows
    args,_=inputs();args['times']=[i*.5 for i in range(7)]
    args['observed']=[args['observed'][0]]*7
    args['initial_quaternions']=[[[1.,0.,0.,0.]]*3]*7
    args['initial_roots']=[[0.,0.,0.]]*7
    group=shared();group.minimum_total=200.;group.maximum_total=250.
    p=_native.RelativeTwistPrior();p.parent=0;p.child=1;p.scale=.5
    result=fit_windows(dict(**args,shared_axial_length=group,relative_twist_priors=[p]))
    total=np.sum(result.lengths,axis=1)
    assert np.all(total>=200.-1e-8) and np.all(total<=250.+1e-8)
