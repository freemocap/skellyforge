"""The experiment must not bridge absences or display a partial rigid segment."""
import numpy as np
import pytest

from experiments.generators.partial_observations import experiment, json_ready


@pytest.mark.parametrize('kind', ['single_point', 'whole_segment', 'absence', 'ends', 'never_seen'])
def test_gap_filled_inputs_and_complete_connected_output(kind):
    case = experiment(kind)
    fitted = case['fitted'].reshape(41, 3, 8, 3)
    visible = np.isfinite(fitted).all(axis=-1)
    # Every modeled segment and every landmark is present together, or all absent.
    assert np.all(visible == visible[:, :1, :1])
    for start, stop in case['report']['blank_spans']:
        assert np.isnan(case['filled'][start:stop]).all()
        assert np.isnan(fitted[start:stop]).all()
    for start, stop in case['report']['active_spans']:
        assert visible[start:stop].all()
        # All eight points share one rigid segment transform; the longest edge is 160 mm.
        np.testing.assert_allclose(np.linalg.norm(fitted[start:stop, :, 1] - fitted[start:stop, :, 0], axis=-1), 160., atol=1e-7)
    if kind == 'whole_segment':
        assert np.isnan(case['raw'][15:23, 8:16]).all()
        assert np.isfinite(case['filled'][15:23, 8:16]).all()
    if kind == 'absence':
        assert [(run['start'], run['stop']) for run in case['runs']] == [(0, 17), (25, 41)]
    if kind == 'never_seen':
        assert np.isnan(case['filled'][:, 16:]).all()
        assert np.isfinite(case['fitted'][:, 16:]).all()
    assert json_ready(np.array([np.nan])) == [None]
