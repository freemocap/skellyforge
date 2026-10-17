import numpy as np
import pytest

from skellyforge.core.trajectories import fill_trajectory_gaps


def test_timestamp_interpolation_preserves_measurements_and_provenance():
    times = np.array([0., .03, .1, .2, .4, .5, .6])
    points = np.ones((7, 3, 3))
    points[:, 0] = times[:, None] * [1., 2., 3.]
    points[3:5, 0] = np.nan
    points[:, 2] = np.nan
    points[2, 2] = 999.  # Unsupported singleton.
    before = points.copy()
    filled, report = fill_trajectory_gaps(points=points, timestamps_s=times)
    np.testing.assert_allclose(filled[:, 0], times[:, None] * [1., 2., 3.])
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(report.original_support(filled), np.isfinite(before).all(axis=-1))
    assert not report.measured_support(filled)[3:5, 0].any()
    assert report.filled_spans == ((0, 3, 5),)
    assert report.discarded_spans == ((2, 2, 3),)
    assert report.unsupported_keypoint_indices == (2,)
    assert report.active_spans == ((0, 7),)
    assert report.to_dict()['algorithm_version'] == 4


def test_blank_frames_split_entry_exit_and_reentry_without_crossing_absence():
    times = np.arange(21) / 10
    points = np.ones((21, 2, 3))
    points[:2] = points[8:12] = points[19:] = np.nan
    points[12:19, 0] = 100.
    points[7, 0] = np.nan  # End of first appearance, not an interior gap.
    points[12, 0] = np.nan  # Beginning of second appearance.
    filled, report = fill_trajectory_gaps(points=points, timestamps_s=times)
    assert report.active_spans == ((2, 8), (12, 19))
    assert report.blank_spans == ((0, 2), (8, 12), (19, 21))
    assert np.isnan(filled[8:12]).all()
    assert np.isnan(filled[[7, 12], 0]).all()


def test_single_blank_frame_is_not_interpolated():
    points = np.ones((11, 2, 3)); points[5] = np.nan
    filled, report = fill_trajectory_gaps(points=points, timestamps_s=np.arange(11) / 10)
    assert np.isnan(filled[5]).all()
    assert report.active_spans == ((0, 5), (6, 11))


def test_never_observed_and_missing_ends_remain_missing():
    points = np.ones((11, 3, 3)); points[:, 2] = np.nan
    points[:3, 0] = points[8:, 0] = np.nan
    filled, report = fill_trajectory_gaps(points=points, timestamps_s=np.arange(11) / 10)
    np.testing.assert_array_equal(filled, points)
    assert report.unsupported_keypoint_indices == (2,)
    assert report.filled_spans == ()


@pytest.mark.parametrize('rate', [6, 30, 60, 120])
def test_support_rule_is_in_seconds(rate):
    times = np.arange(rate + 1) / rate
    points = np.ones((len(times), 1, 3))
    filled, report = fill_trajectory_gaps(points=points, timestamps_s=times)
    np.testing.assert_array_equal(filled, points)
    assert report.discarded_spans == ()


def test_entirely_absent_person_has_no_active_interval():
    filled, report = fill_trajectory_gaps(points=np.full((5, 2, 3), np.nan), timestamps_s=np.arange(5.))
    assert np.isnan(filled).all()
    assert report.active_spans == ()
    assert report.blank_spans == ((0, 5),)


@pytest.mark.parametrize('bad', ['time', 'partial_xyz', 'infinity'])
def test_invalid_input_is_rejected(bad):
    points = np.ones((5, 2, 3)); times = np.arange(5.)
    if bad == 'time': times[2] = times[1]
    if bad == 'partial_xyz': points[2, 0, 0] = np.nan
    if bad == 'infinity': points[2, 0] = np.inf
    with pytest.raises(ValueError):
        fill_trajectory_gaps(points=points, timestamps_s=times)
