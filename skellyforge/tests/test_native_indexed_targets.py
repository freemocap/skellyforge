"""A missing keypoint removes its residual, not its model landmark."""
import numpy as np
import pytest
from skellyforge import _native


def problem():
    local = [[0.,0.,0.], [100.,0.,0.], [0.,100.,0.], [0.,0.,100.]]
    slots = [[0,1,2,3], [3,0,2], [2,1,0,3]]
    return dict(local=[local], observed=[[[local[j] for j in ids]] for ids in slots],
                observation_indices=[[ids] for ids in slots], times=[0.,.1,.2],
                parent_indices=[], parent_attachments=[], child_attachments=[],
                initial_quaternions=[[[1.,0.,0.,0.]]]*3, initial_roots=[[3.,-2.,1.]]*3,
                position_scale=1., linear_acceleration_scale=100., angular_acceleration_scale=10.)


def test_partial_reordered_targets_preserve_model_correspondence():
    args=problem()
    result=_native.fit_chain_sequence(**args)
    assert result.converged
    np.testing.assert_allclose(result.roots, 0., atol=1e-6)
    assert result.landmark_cost < 1e-10
    # Eleven supplied XYZ targets and two temporal residual blocks.
    assert result.residual_blocks == 13
    assert len(args['local'][0]) == 4


@pytest.mark.parametrize('indices', [[0,0,2], [0,1,4], [0,-1,2]])
def test_invalid_correspondences_rejected(indices):
    args=problem(); args['observation_indices'][1][0]=indices
    with pytest.raises(ValueError, match='invalid or repeated'):
        _native.fit_chain_sequence(**args)


def test_frame_without_keypoints_retains_pose_parameter_blocks():
    args=problem(); args['observed'][1]=[[]]; args['observation_indices'][1]=[[]]
    result=_native.fit_chain_sequence(**args)
    assert result.parameter_blocks == 6
    assert result.residual_blocks == 10
    assert len(result.quaternions) == 3
    np.testing.assert_allclose(result.roots, 0., atol=1e-6)
