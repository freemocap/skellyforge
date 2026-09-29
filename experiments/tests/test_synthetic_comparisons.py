"""Synthetic comparison generators retain their numerical and display contracts."""
import numpy as np
from experiments.generators.solver_chain_experiment import chain_experiment
from experiments.generators.solver_tree_experiment import tree_experiment
from experiments.generators.solver_torso_experiment import torso_experiment


def test_branching_exact_attachments_and_gap_metrics():
    run=chain_experiment(noise=1, gap=5, branching=True)
    method=run["methods"]["temporal"]
    assert method["summary"]["Ceres parameter blocks"]==164
    assert method["summary"]["Ceres residual blocks"]==1100
    for frame in method["frames"]:
        assert frame["converged"]
        assert frame["diagnostics"]["First attachment separation (mm)"]<1e-9
        assert frame["diagnostics"]["Second attachment separation (mm)"]<1e-9
        assert all(np.isfinite(b["angular_error_degrees"]) for b in frame["bodies"])
    assert method["frames"][20]["bodies"][1]["observed"]==[None]*8

def test_five_segment_tree_connections():
    run=tree_experiment(noise=1.)
    method=run['methods']['temporal']
    assert method['summary']['Ceres parameter blocks']==126
    assert method['summary']['Ceres residual blocks']==954
    for frame in method['frames']:
        assert frame['converged']
        assert max(v for k,v in frame['diagnostics'].items() if 'equation error' in k)<1e-9
        assert max(b['truth_rms'] for b in frame['bodies'])<3.

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


def test_displacement_and_gap_use_identical_observations_in_both_problems():
    from experiments.generators.solver_displacement_experiment import displacement_experiment

    run = displacement_experiment(noise=1, extension=20, gap=5)
    fixed = run["methods"]["fixed"]
    fitted = run["methods"]["displacement"]
    assert fitted["summary"]["Ceres parameter blocks"] == 205
    assert fitted["summary"]["Ceres residual blocks"] == 1180
    assert fixed["summary"]["Ceres residual blocks"] == 1100
    for i, (a, b) in enumerate(zip(fixed["frames"], fitted["frames"])):
        assert a["converged"] and b["converged"]
        assert [x["observed"] for x in a["bodies"]] == [x["observed"] for x in b["bodies"]]
        assert b["diagnostics"]["Second linkage equation error (mm)"] < 1e-9
        assert abs(b["displacement"]) <= 40
        if i in run["gap_frames"]:
            assert b["bodies"][1]["observed"] == [None] * 8
            assert b["diagnostics"]["Middle landmark residual blocks"] == 0
    assert fitted["summary"]["All landmark RMS vs known (mm)"] < fixed["summary"]["All landmark RMS vs known (mm)"]
