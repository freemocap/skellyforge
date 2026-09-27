"""Warm starts, immutable boundary states, and one globally evaluated trajectory."""
import numpy as np
import pytest
from skellyforge import _native
from skellyforge.tests.test_native_axial import fixture
from scripts.solver_window_sequence import fit_windows,frame_weights,refine_window_result


def seeded():
    args,*_=fixture();args['free_axial_lengths']=True
    start=_native.fit_chain_sequence(**args)
    args['initial_quaternions']=start.initial_quaternions
    args['initial_roots']=[f[0] for f in start.initial_translations]
    return args,start


def options_from(result):
    options=_native.ChainSolveOptions();options.initial_lengths=result.lengths
    options.initial_linkage_displacements=result.linkage_displacements
    return options


def test_strict_initial_window_preserves_its_committed_pose():
    args,_=seeded()
    baseline=fit_windows(args,function_tolerance=1e-6)
    candidate=fit_windows(args,function_tolerance=1e-4,initial_function_tolerance=1e-6)
    np.testing.assert_array_equal(candidate.quaternions[0],baseline.quaternions[0])
    np.testing.assert_array_equal(candidate.lengths[0],baseline.lengths[0])
    np.testing.assert_array_equal(candidate.roots[0],baseline.roots[0])
    assert candidate.processing['initial_function_tolerance']==1e-6


def test_evaluation_and_fixed_prefix_preserve_all_supplied_parameters():
    args,start=seeded();options=options_from(start);options.evaluate_only=True
    args.update(initial_quaternions=start.quaternions,initial_roots=start.roots)
    result=_native.fit_chain_sequence(**args,solve_options=options)
    np.testing.assert_array_equal(result.lengths,start.lengths)
    np.testing.assert_array_equal(result.quaternions,start.quaternions)
    np.testing.assert_array_equal(result.roots,start.roots)
    assert result.costs[-1]==pytest.approx(start.costs[-1],abs=1e-8)
    assert result.iterations==0 and not result.converged
    options.evaluate_only=False;options.fixed_prefix_frames=2
    changed=np.array(start.lengths);changed[:2,1]+=5;options.initial_lengths=changed.tolist()
    result=_native.fit_chain_sequence(**args,solve_options=options)
    np.testing.assert_array_equal(np.array(result.lengths)[:2],changed[:2])
    np.testing.assert_array_equal(result.quaternions[:2],start.quaternions[:2])
    np.testing.assert_array_equal(result.roots[:2],start.roots[:2])
    assert 'Time (in seconds)' in result.full_report


@pytest.mark.parametrize('window',[3,5,7])
def test_windows_cover_every_frame_keep_geometry_and_global_score(window):
    args,start=seeded();result=fit_windows(args,active_frames=window)
    n=len(args['times']);committed=[]
    for w in result.processing['windows']:
        assert w['active_end']-w['active_start']+1==window
        assert w['active_start']-w['fixed_start']==min(2,w['active_start'])
        committed.extend(range(w['active_start'],w['committed_end']+1))
    assert committed==list(range(n))
    np.testing.assert_allclose(result.lengths,start.lengths,atol=.02)
    np.testing.assert_allclose(np.linalg.norm(result.quaternions,axis=2),1,atol=1e-10)
    options=options_from(result);options.evaluate_only=True
    evaluated=_native.fit_chain_sequence(**{**args,'initial_quaternions':result.quaternions,'initial_roots':result.roots,'solve_options':options})
    assert result.costs[-1]==pytest.approx(evaluated.costs[-1],abs=1e-10)
    refined=refine_window_result(args,result)
    np.testing.assert_array_equal(refined.initial_quaternions,result.quaternions)
    assert refined.costs[0]==pytest.approx(result.costs[-1],abs=1e-8)
    assert refined.costs[-1]<=result.costs[-1]+1e-8


def test_full_sequence_weights_are_retained_and_invalid_options_rejected():
    args,_=seeded();weights=frame_weights(args['times'])
    result=fit_windows(args)
    np.testing.assert_array_equal(result.processing['frame_weights'],weights)
    for field,value in [('frame_weights',[1.]),('frame_weights',[-1.]*len(weights)),
                        ('fixed_prefix_frames',-1),('initial_lengths',[[1.]]),('max_iterations',-1),
                        ('function_tolerance',float('nan')),('function_tolerance',-1.)]:
        options=_native.ChainSolveOptions();setattr(options,field,value)
        with pytest.raises(ValueError):_native.fit_chain_sequence(**args,solve_options=options)


def test_one_window_matches_whole_sequence_and_missing_keypoints_stay_absent():
    args,_=seeded()
    direct=_native.fit_chain_sequence(**args)
    single=fit_windows(args,active_frames=len(args['times']))
    np.testing.assert_allclose(single.lengths,direct.lengths,atol=1e-8)
    assert single.costs[-1]==pytest.approx(direct.costs[-1],abs=1e-9)
    args['observation_indices']=[[list(range(len(points))) for points in frame] for frame in args['observed']]
    for i in range(8,11):
        args['observed'][i]=[[] for _ in args['local']]
        args['observation_indices'][i]=[[] for _ in args['local']]
    result=fit_windows(args)
    assert np.isfinite(result.quaternions).all() and np.isfinite(result.lengths).all()
    assert all(sum(map(len,args['observed'][i]))==0 for i in range(8,11))


def test_fixed_history_preserves_shoulder_displacement_parameters():
    args,start=seeded();args['relaxed_linkage_children']=[3]
    options=options_from(start);options.fixed_prefix_frames=2
    delta=np.array(options.initial_linkage_displacements);delta[:2,3]=[2.,3.,4.]
    options.initial_linkage_displacements=delta.tolist()
    result=_native.fit_chain_sequence(**args,solve_options=options)
    np.testing.assert_array_equal(np.array(result.linkage_displacements)[:2],delta[:2])
