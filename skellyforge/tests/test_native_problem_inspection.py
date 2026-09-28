"""Inspect Ceres' registered problem, without a parallel graph definition."""
import numpy as np
import pytest
from skellyforge import _native
from skellyforge.tests.test_native_window_sequence import seeded
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows


def test_inspection_does_not_change_solution_and_accounts_for_every_block():
    args,_=seeded()
    options=_native.ChainSolveOptions();options.landmark_huber_scale_mm=30.
    plain=_native.fit_chain_sequence(**args,solve_options=options)
    assert plain.problem_initial is None and plain.problem_final is None
    options.inspect_problem=True
    inspected=_native.fit_chain_sequence(**args,solve_options=options)
    np.testing.assert_array_equal(inspected.quaternions,plain.quaternions)
    np.testing.assert_array_equal(inspected.roots,plain.roots)
    np.testing.assert_array_equal(inspected.lengths,plain.lengths)
    np.testing.assert_array_equal(inspected.costs,plain.costs)
    initial,final=inspected.problem_initial,inspected.problem_final
    assert len(final.parameters)==inspected.parameter_blocks
    assert len(final.residuals)==inspected.residual_blocks
    assert sum(r.cost for r in final.residuals)==pytest.approx(inspected.costs[-1],abs=1e-8)
    assert sum(r.cost for r in initial.residuals)==pytest.approx(inspected.costs[0],abs=1e-8)
    assert [r.parameter_ids for r in initial.residuals]==[r.parameter_ids for r in final.residuals]
    assert all(p.quantity and 0 <= p.frame < len(args["times"]) for p in final.parameters)
    assert all(r.purpose for r in final.residuals)
    for p in final.parameters:
        if p.quantity == "World quaternion (wxyz)":
            np.testing.assert_array_equal(p.values, inspected.quaternions[p.frame][p.segment])
        elif p.quantity == "Root translation (mm)":
            np.testing.assert_array_equal(p.values, inspected.roots[p.frame])
    ids={p.id for p in final.parameters}
    assert all(set(r.parameter_ids)<=ids for r in final.residuals)
    quaternions=[p for p in final.parameters if p.manifold_type]
    assert quaternions and all((p.ambient_size,p.tangent_size)==(4,3) for p in quaternions)
    robust=[r for r in final.residuals if r.loss_type]
    assert robust and all('HuberLoss' in r.loss_type for r in robust)
    assert all(r.cost<=.5*np.dot(r.values,r.values)+1e-8 for r in robust)


def test_constant_history_and_length_bounds_are_read_from_problem():
    args,_=seeded();options=_native.ChainSolveOptions()
    options.inspect_problem=True;options.fixed_prefix_frames=2;options.fixed_length_segments=[1]
    fit=_native.fit_chain_sequence(**args,solve_options=options)
    initial=fit.problem_initial;final=fit.problem_final
    fixed=[p for p in initial.parameters if p.constant]
    assert fixed and any(p.manifold_type for p in fixed)
    for p in fixed:
        np.testing.assert_array_equal(p.values,final.parameters[p.id].values)
    lengths=[p for p in final.parameters if p.ambient_size==1]
    assert lengths and all(p.lower_bounds==[0.] for p in lengths)
    assert sum(p.constant for p in lengths)==len(args['times'])+2


def test_new_registered_prior_appears_without_inspector_changes():
    args,_=seeded();options=_native.ChainSolveOptions();options.inspect_problem=True
    baseline=_native.fit_chain_sequence(**args,solve_options=options)
    prior=_native.LandmarkPositionPrior();prior.segment=2;prior.local_point=[0.,0.,0.]
    prior.targets=[[0.,0.,0.]]*len(args['times']);prior.scale=20.
    options.landmark_position_priors=[prior]
    fit=_native.fit_chain_sequence(**args,solve_options=options)
    assert len(fit.problem_final.residuals)==len(baseline.problem_final.residuals)+len(args['times'])
    assert sum(r.cost for r in fit.problem_final.residuals)==pytest.approx(fit.costs[-1],abs=1e-8)


def test_selected_window_keeps_its_actual_problem_not_final_sequence_reconstruction():
    args,_=seeded();plain=fit_windows(args)
    fit=fit_windows(args,inspect_windows=[0,2])
    np.testing.assert_array_equal(fit.quaternions,plain.quaternions)
    assert set(fit.inspected_windows)=={0,2}
    for index,snapshot in fit.inspected_windows.items():
        report=fit.processing['windows'][index]
        assert len(snapshot['final'].parameters)==report['parameter_blocks']
        assert len(snapshot['final'].residuals)==report['residual_blocks']
        assert sum(r.cost for r in snapshot['initial'].residuals)==pytest.approx(report['initial_cost'],abs=1e-8)
        assert sum(r.cost for r in snapshot['final'].residuals)==pytest.approx(report['final_cost'],abs=1e-8)
    assert not any(p.constant for p in fit.inspected_windows[0]['final'].parameters)
    assert any(p.constant for p in fit.inspected_windows[2]['final'].parameters)
