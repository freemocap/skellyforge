"""Sparse rigid torso: declared observations, exact connections, explicit priors."""
import numpy as np
import pytest
from skellyforge import _native
from test_support.torso import model, inputs, initialization


def arguments():
    m=model();times,observed,_=inputs(m,0,0)
    q,roots=initialization(m,observed)
    # Three frames are enough for residual accounting and validation tests.
    return dict(local=m['local'],observed=observed[:3],parent_attachments=m['parent_attachments'],child_attachments=m['child_attachments'],parent_indices=m['parents'],times=times[:3].tolist(),
        position_scale=10.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.,initial_quaternions=q[:3],initial_roots=roots[:3],rest_relative_quaternions=m['relative'],rest_pose_scale=2.)


def test_sparse_prior_cost_accounting_and_quaternion_sign_invariance():
    args=arguments();a=_native.fit_chain_sequence(**args)
    flipped={**args,'initial_quaternions':(-np.array(args['initial_quaternions'])).tolist(), 'rest_relative_quaternions':(-np.array(args['rest_relative_quaternions'])).tolist()}
    b=_native.fit_chain_sequence(**flipped)
    assert a.converged and b.converged
    assert a.parameter_blocks==18
    assert a.residual_blocks==30  # 12 landmark + 12 relative pose + 6 temporal
    assert a.costs[-1]==pytest.approx(a.landmark_cost+a.relative_pose_cost+a.root_acceleration_cost+sum(a.angular_acceleration_costs),rel=1e-7)
    np.testing.assert_allclose(a.translations,b.translations,atol=1e-7)
    assert a.costs[-1]==pytest.approx(b.costs[-1],abs=1e-10)




@pytest.mark.parametrize('field,value', [('initial_roots',[]),('rest_pose_scale',0.),('rest_relative_quaternions',[[1,0,0,0]]),('initial_quaternions',[[[2,0,0,0]]*5]*3)])
def test_invalid_sparse_initialization_and_priors_rejected(field,value):
    args=arguments();args[field]=value
    with pytest.raises(ValueError): _native.fit_chain_sequence(**args)
