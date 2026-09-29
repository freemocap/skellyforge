"""Model output exists independently of measured targets and uses fitted geometry."""
from types import SimpleNamespace

import numpy as np

from skellyforge.core.skeleton.fitting.human import HumanFit


def test_all_landmarks_share_segment_transform_and_axial_scale_without_observations():
    # A 90-degree rotation about X moves local +Z onto world -Y.
    q = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
    sequence = SimpleNamespace(
        quaternions=[[q], [q]], translations=[[[10., 20., 30.]], [[40., 50., 60.]]],
        lengths=[[4.], [6.]],
    )
    model = dict(display_names=[['origin', 'tip', 'off_axis']], references=[2.],
                 display=[np.array([[0., 0., 0.], [0., 0., 2.], [1., 2., 1.]])])
    before = model['display'][0].copy()
    fit = HumanFit(model, dict(observed=[[[]], [[]]]), sequence)
    points = fit.landmark_positions()
    assert set(points) == {'origin', 'tip', 'off_axis'}
    np.testing.assert_allclose(points['origin'], [[10., 20., 30.], [40., 50., 60.]])
    np.testing.assert_allclose(points['tip'], [[10., 16., 30.], [40., 44., 60.]])
    np.testing.assert_allclose(points['off_axis'], [[11., 18., 32.], [41., 47., 62.]])
    np.testing.assert_array_equal(model['display'][0], before)
