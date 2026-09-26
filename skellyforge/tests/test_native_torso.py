"""Sparse rigid torso: declared observations, exact connections, explicit priors."""
import numpy as np
import pytest
from skellyforge import _native
from scripts.solver_torso_experiment import model, inputs, initialization, torso_experiment


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


def test_four_landmarks_same_initialization_and_exact_rigid_connections():
    run=torso_experiment(noise=1,motion=0)
    a,b=(run['methods'][n] for n in ['no_prior','rest_prior'])
    assert b['frames'][0]['converged']
    assert a['summary']['Ceres residual blocks']==198
    assert b['summary']['Ceres residual blocks']==282
    for fa,fb in zip(a['frames'],b['frames']):
        assert sum(p is not None for body in fb['bodies'] for p in body['observed'])==4
        assert max(v for k,v in fb['diagnostics'].items() if 'attachment error' in k)<1e-9
        for ba,bb in zip(fa['bodies'],fb['bodies']):
            assert ba['observed']==bb['observed']
            assert ba['initial_quaternion']==bb['initial_quaternion']
            assert ba['initial_translation']==bb['initial_translation']
            local=np.array(bb['local']);fitted=np.array(bb['fitted'])
            np.testing.assert_allclose(np.linalg.norm(local[:,None]-local,axis=-1),np.linalg.norm(fitted[:,None]-fitted,axis=-1),atol=1e-9)


@pytest.mark.parametrize('field,value', [('initial_roots',[]),('rest_pose_scale',0.),('rest_relative_quaternions',[[1,0,0,0]]),('initial_quaternions',[[[2,0,0,0]]*5]*3)])
def test_invalid_sparse_initialization_and_priors_rejected(field,value):
    args=arguments();args[field]=value
    with pytest.raises(ValueError): _native.fit_chain_sequence(**args)
