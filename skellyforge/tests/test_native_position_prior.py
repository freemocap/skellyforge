"""Position preferences act through connected FK and retain distinct costs."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.tests.test_native_window_sequence import seeded
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows, frame_weights, refine_window_result
from test_support.geometry import axial_points


def test_missing_position_evidence_is_absent_from_the_objective():
    args, start = seeded()
    prior = _native.LandmarkPositionPrior()
    prior.segment = 1
    prior.local_point = [0., 0., 20.]
    prior.scale = 2.
    targets = (np.asarray(start.translations)[:, 1] + [5., 3., 20.]).tolist()
    active = np.arange(len(targets)) % 3 == 0
    prior.targets = [target if available else None for target, available in zip(targets, active)]
    args['landmark_position_priors'] = [prior]
    fit = fit_windows(args)
    for result in (fit, refine_window_result(args, fit)):
        rotations = Rotation.from_quat(np.asarray(result.quaternions)[:, 1], scalar_first=True)
        local = np.tile(prior.local_point, (len(targets), 1))
        local[:, 2] *= np.asarray(result.lengths)[:, 1] / args['axial_reference_lengths'][1]
        modeled = rotations.apply(local) + np.asarray(result.translations)[:, 1]
        expected = .5 * np.sum(frame_weights(args['times'])[active] *
            np.sum(((modeled[active] - np.asarray(targets)[active]) / prior.scale) ** 2, axis=1))
        assert result.position_prior_cost == pytest.approx(expected, rel=1e-8, abs=1e-8)
    prior.targets = [None] * len(targets)
    empty = fit_windows(args)
    baseline = fit_windows({key: value for key, value in args.items() if key != 'landmark_position_priors'})
    assert empty.position_prior_cost == 0.
    np.testing.assert_array_equal(empty.quaternions, baseline.quaternions)


def test_position_prior_survives_window_slicing_and_refinement():
    args,start=seeded()
    p=_native.LandmarkPositionPrior();p.segment=1;p.local_point=[0.,0.,20.];p.scale=2.
    p.targets=(np.asarray(start.translations)[:,1]+[5.,3.,20.]).tolist()
    args['landmark_position_priors']=[p]
    fit=fit_windows(args)
    for result in (fit,refine_window_result(args,fit)):
        rotations=Rotation.from_quat(np.asarray(result.quaternions)[:,1],scalar_first=True)
        points=np.tile(p.local_point,(len(args['times']),1))
        points[:,2]*=np.asarray(result.lengths)[:,1]/args['axial_reference_lengths'][1]
        fitted=rotations.apply(points)+np.asarray(result.translations)[:,1]
        expected=.5*np.sum(frame_weights(args['times'])*np.sum(((fitted-p.targets)/p.scale)**2,axis=1))
        assert result.position_prior_cost==pytest.approx(expected,rel=1e-8,abs=1e-8)
        assert result.position_prior_cost>0


@pytest.mark.parametrize('bad', ['segment','scale','targets'])
def test_position_prior_rejects_invalid_definition(bad):
    args,_=seeded();p=_native.LandmarkPositionPrior()
    p.segment=0;p.targets=[[0.,0.,0.]]*len(args['times'])
    if bad=='segment':p.segment=-1
    if bad=='scale':p.scale=0.
    if bad=='targets':p.targets=[]
    options=_native.ChainSolveOptions();options.landmark_position_priors=[p]
    with pytest.raises(ValueError,match='Invalid landmark position prior'):
        _native.fit_chain_sequence(**args,solve_options=options)


@pytest.mark.parametrize('time_scale',[1.,5.])
def test_huber_transition_is_in_mm_not_weighted_residual_units(time_scale):
    args,start=seeded()
    args['times']=(np.asarray(args['times'])*time_scale).tolist()
    args['observed'][2][0][0][0]+=1000.
    options=_native.ChainSolveOptions();options.evaluate_only=True
    options.initial_lengths=start.lengths;options.initial_linkage_displacements=start.linkage_displacements
    options.landmark_huber_scale_mm=30.
    args.update(initial_quaternions=start.quaternions,initial_roots=start.roots)
    result=_native.fit_chain_sequence(**args,solve_options=options)
    expected=0.
    for i,weight in enumerate(frame_weights(args['times'])):
        for b,local in enumerate(args['local']):
            fitted=Rotation.from_quat(result.quaternions[i][b],scalar_first=True).apply(
                axial_points(local,args['axial_reference_lengths'][b],result.lengths[i][b]))+result.translations[i][b]
            d=np.linalg.norm(fitted-args['observed'][i][b],axis=1)
            rho=np.where(d<=30.,d*d,60.*d-900.)
            expected+=.5*weight*np.sum(rho)/args['position_scale']**2
    assert result.landmark_cost==pytest.approx(expected,rel=1e-9)


def test_extreme_child_target_does_not_drag_root_with_robust_loss():
    args,start=seeded()
    clean=_native.fit_chain_sequence(**args)
    for frame in args['observed']:
        frame[-1][0][2]-=1000.
    ordinary=_native.fit_chain_sequence(**args)
    options=_native.ChainSolveOptions();options.landmark_huber_scale_mm=30.
    robust=_native.fit_chain_sequence(**args,solve_options=options)
    ordinary_error=np.linalg.norm(np.asarray(ordinary.roots)-clean.roots)
    robust_error=np.linalg.norm(np.asarray(robust.roots)-clean.roots)
    assert robust_error<ordinary_error/2
