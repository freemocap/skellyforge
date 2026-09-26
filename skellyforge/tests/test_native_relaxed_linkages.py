"""General linkage displacement: real Ceres solves, including descendants."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native


def solve(relaxed=(1,), scale=100., rotate=False, varying=False):
    local=np.array([[0,0,0],[30,0,0],[0,40,0],[0,0,50.]])
    # Root branches to 1 and 3; segment 2 descends from relaxed linkage 1.
    parents=[0,1,0];attachments=np.array([[100.,0,0],[100,0,0],[0,100,0]])
    translations=np.array([[0,0,0],[100,12,0],[200,12,0],[0,100,0.]])
    rotation=Rotation.from_euler('xyz',[.3,-.4,.6]) if rotate else Rotation.identity()
    observed=rotation.apply((local[None,:,:]+translations[:,None,:]).reshape(-1,3)).reshape(4,4,3)
    frames=np.repeat(observed[None],3,axis=0)
    if varying:
        frames[1,1:3]+=rotation.apply([0,8,0])
    return _native.fit_chain_sequence(local=[local.tolist()]*4,observed=frames.tolist(),
        parent_attachments=attachments.tolist(),child_attachments=[[0,0,0]]*3,parent_indices=parents,
        times=[0,.1,.2],position_scale=1.,linear_acceleration_scale=3000.,angular_acceleration_scale=20.,
        relaxed_linkage_children=list(relaxed),linkage_scale=scale,linkage_acceleration_scale=300.)


def test_displacement_recovers_translation_and_preserves_other_joints():
    result=solve();assert result.converged
    assert result.parameter_blocks==18
    assert result.residual_blocks==57
    np.testing.assert_allclose(np.array(result.linkage_displacements)[:,1],[ [0,12,0] ]*3,atol=.05)
    for frame in range(3):
        t=np.array(result.translations[frame]);q=Rotation.from_quat(result.quaternions[frame],scalar_first=True)
        np.testing.assert_allclose(t[2],t[1]+q[1].apply([100,0,0]),atol=1e-9)
        np.testing.assert_allclose(t[3],t[0]+q[0].apply([0,100,0]),atol=1e-9)
    assert result.linkage_prior_cost>0
    assert result.linkage_acceleration_cost<1e-12
    total=result.landmark_cost+result.root_acceleration_cost+sum(result.angular_acceleration_costs)+result.linkage_prior_cost+result.linkage_acceleration_cost
    assert result.costs[-1]==pytest.approx(total,abs=1e-9)


def test_displacement_is_parent_local_and_strength_controls_separation():
    a,b=solve(),solve(rotate=True)
    np.testing.assert_allclose(a.linkage_displacements,b.linkage_displacements,atol=.01)
    tight=solve(scale=.01)
    assert np.linalg.norm(np.array(tight.linkage_displacements)[:,1],axis=1).max()<.01
    exact=solve(relaxed=())
    assert exact.parameter_blocks==15
    assert exact.linkage_prior_cost==0
    np.testing.assert_array_equal(exact.linkage_displacements,np.zeros((3,4,3)))


def test_local_displacement_temporal_cost_matches_returned_parameters():
    result=solve(varying=True)
    delta=np.asarray(result.linkage_displacements)[:,1]
    velocity_difference=(delta[2]-delta[1])/.1-(delta[1]-delta[0])/.1
    residual=velocity_difference/(300.*np.sqrt(.1))
    assert result.linkage_acceleration_cost>0
    assert result.linkage_acceleration_cost==pytest.approx(.5*np.dot(residual,residual),rel=1e-10)


@pytest.mark.parametrize('children',[(0,),(4,),(1,1),(-1,)])
def test_invalid_linkages_fail(children):
    with pytest.raises(ValueError,match='unique non-root'):
        solve(relaxed=children)
