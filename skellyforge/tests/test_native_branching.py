"""Fixed branches share a parent pose, never a sibling quaternion."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts.solver_chain_experiment import chain_inputs, chain_experiment


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


def test_branch_order_does_not_change_fit():
    local, parent, child, times, _, records=chain_inputs(noise=1,gap=0,branching=True)
    observed=np.array([r["observed"] for r in records])
    def solve(order):
        return _native.fit_chain_sequence(local=[local.tolist()]*3,
            observed=observed[:,order].tolist(),parent_attachments=parent[np.array(order[1:])-1].tolist(),
            child_attachments=child[np.array(order[1:])-1].tolist(),times=times.tolist(),
            position_scale=10.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.,parent_indices=[0,0])
    a,b=solve([0,1,2]),solve([0,2,1])
    assert a.converged and b.converged
    np.testing.assert_allclose(a.translations,np.array(b.translations)[:,[0,2,1]],atol=1e-4)
    assert a.costs[-1]==pytest.approx(b.costs[-1],rel=1e-7)
    for qa,qb in zip(np.array(a.quaternions).reshape(-1,4),np.array(b.quaternions)[:,[0,2,1]].reshape(-1,4)):
        assert (Rotation.from_quat(qa,scalar_first=True).inv()*Rotation.from_quat(qb,scalar_first=True)).magnitude()<1e-5


def test_five_segment_tree_connections():
    from scripts.solver_tree_experiment import tree_experiment
    run=tree_experiment(noise=1.)
    method=run['methods']['temporal']
    assert method['summary']['Ceres parameter blocks']==126
    assert method['summary']['Ceres residual blocks']==954
    for frame in method['frames']:
        assert frame['converged']
        assert max(v for k,v in frame['diagnostics'].items() if 'equation error' in k)<1e-9
        assert max(b['truth_rms'] for b in frame['bodies'])<3.


@pytest.mark.parametrize('parents', [[1,0,2,2], [0,1,4,2], [0,-1,2,2], [0,1]])
def test_tree_rejects_invalid_topology(parents):
    from scripts.solver_tree_experiment import tree_inputs
    local,_,parent,child,times,records=tree_inputs()
    with pytest.raises(ValueError):
        _native.fit_chain_sequence(local=[local.tolist()]*5,observed=[r['observed'].tolist() for r in records],
            parent_attachments=parent.tolist(),child_attachments=child.tolist(),parent_indices=parents,times=times.tolist(),
            position_scale=10.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.)
