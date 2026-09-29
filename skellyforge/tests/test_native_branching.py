"""Fixed branches share a parent pose, never a sibling quaternion."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from test_support.chain import chain_inputs




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




@pytest.mark.parametrize('parents', [[1,0,2,2], [0,1,4,2], [0,-1,2,2], [0,1]])
def test_tree_rejects_invalid_topology(parents):
    from test_support.tree import tree_inputs
    local,_,parent,child,times,records=tree_inputs()
    with pytest.raises(ValueError):
        _native.fit_chain_sequence(local=[local.tolist()]*5,observed=[r['observed'].tolist() for r in records],
            parent_attachments=parent.tolist(),child_attachments=child.tolist(),parent_indices=parents,times=times.tolist(),
            position_scale=10.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.)
