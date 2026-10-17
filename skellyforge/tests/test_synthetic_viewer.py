"""Closed-form viewer regressions retained independently of the optimizer."""
import numpy as np
from skellyforge.tools.viewer import synthetic


def test_torso_and_leg_motion_reaches_synthetic_observations():
    stationary, moving = {}, {}
    synthetic._build_data(capture=stationary)
    data = synthetic._build_data(torso=True, legs=True, capture=moving)
    assert data['motion']['torso'] and data['motion']['legs']
    for name in ('thoracic', 'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg'):
        fixed = np.asarray([frame['rotations'][name] for frame in stationary['records']])
        rotations = np.asarray([frame['rotations'][name] for frame in moving['records']])
        np.testing.assert_allclose(fixed, np.broadcast_to(fixed[0], fixed.shape), atol=1e-10)
        assert np.isfinite(rotations).all()
        assert np.max(np.ptp(rotations, axis=0)) > .01, name
    assert len(moving['records']) == data['frame_count']


def test_synthetic_capture_uses_the_displayed_inputs():
    capture = {}
    data = synthetic._build_data(shoulders=True, capture=capture)
    assert len(capture['records']) == data['frame_count']
    assert capture['skeleton'].name
    for record, frame in zip(capture['records'], data['frames'], strict=True):
        for name, position in zip(data['landmarks_meta'], frame['landmarks'], strict=True):
            import numpy as np
            np.testing.assert_allclose(record['points'][name], position, atol=0.0051, rtol=0  # Existing viewer rounds display coordinates to 0.01 mm.
            )
            np.testing.assert_array_equal(record['keypoints']['synthetic:'+name], record['points'][name])
