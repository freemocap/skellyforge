"""Axial deformation belongs in residual geometry and exact FK, not render repair."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts.solver_tree_experiment import tree_inputs
from scripts.solver_axial_geometry import axial_points


def fixture(flexible=True, shortening=True):
    local,parents,pa,ca,times,records=tree_inputs(noise=0)
    references=[0.,80.,80.,0.,0.]
    lengths=[0.,56.,64.,0.,0.] if shortening else references
    observations=[]
    for record in records:
        rotations=record['rotations'];translations=[record['translations'][0]]
        for b,parent in enumerate(parents,1):
            translations.append(translations[parent]+rotations[parent].apply(axial_points(pa[b-1],references[parent],lengths[parent]))-rotations[b].apply(axial_points(ca[b-1],references[b],lengths[b])))
        observations.append([(r.apply(axial_points(local,references[b],lengths[b]))+translations[b]).tolist() for b,r in enumerate(rotations)])
    args=dict(local=[local.tolist()]*5,observed=observations,parent_attachments=pa.tolist(),child_attachments=ca.tolist(),parent_indices=parents,times=times.tolist(),
        position_scale=1.,linear_acceleration_scale=1e6,angular_acceleration_scale=1e6,axial_reference_lengths=references if flexible else [],length_prior_fraction=1e6,length_acceleration_scale=1e6)
    return args,local,references,lengths


def test_recovers_synthetic_lengths_and_keeps_deformed_attachments_exact():
    args,local,refs,lengths=fixture()
    result=_native.fit_chain_sequence(**args)
    assert result.converged
    np.testing.assert_allclose(np.array(result.lengths)[:,1:3],np.tile(lengths[1:3],(21,1)),atol=1e-3)
    for i in range(21):
        rotations=[Rotation.from_quat(q,scalar_first=True) for q in result.quaternions[i]]
        for b,parent in enumerate(args['parent_indices'],1):
            a=rotations[parent].apply(axial_points(args['parent_attachments'][b-1],refs[parent],result.lengths[i][parent]))+result.translations[i][parent]
            c=rotations[b].apply(axial_points(args['child_attachments'][b-1],refs[b],result.lengths[i][b]))+result.translations[i][b]
            np.testing.assert_allclose(a,c,atol=1e-9)
        for b in range(5):
            predicted=rotations[b].apply(axial_points(local,refs[b],result.lengths[i][b]))+result.translations[i][b]
            np.testing.assert_allclose(predicted,args['observed'][i][b],atol=1e-3)
    assert result.parameter_blocks==168
    assert result.residual_blocks==1034
    assert result.costs[-1]==pytest.approx(result.landmark_cost+result.root_acceleration_cost+sum(result.angular_acceleration_costs)+result.length_prior_cost+result.length_acceleration_cost,abs=1e-8)


def test_reference_lengths_reproduce_rigid_geometry():
    args,_,_,_=fixture(shortening=False)
    a=_native.fit_chain_sequence(**args)
    b=_native.fit_chain_sequence(**{**args,'axial_reference_lengths':[]})
    np.testing.assert_allclose(a.translations,b.translations,atol=1e-4)


def test_thoracic_attachment_group_preserves_shape_and_xy():
    points=np.array([[10,20,80],[-10,20,80],[0,0,0]])
    deformed=axial_points(points,80,50)
    np.testing.assert_array_equal(deformed[:,:2],points[:,:2])
    np.testing.assert_array_equal(deformed[0]-deformed[1],points[0]-points[1])
    np.testing.assert_array_equal(deformed[:,2],[50,50,0])


@pytest.mark.parametrize('references',[[0,80],[0,-80,80,0,0],[0,float('nan'),80,0,0]])
def test_invalid_axial_definitions_rejected(references):
    args,*_=fixture()
    with pytest.raises(ValueError):_native.fit_chain_sequence(**{**args,'axial_reference_lengths':references})
