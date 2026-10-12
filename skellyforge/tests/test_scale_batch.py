"""Ragged sample counts and absent evidence must survive grouped reductions."""

import pytest
from skellyforge.core.skeleton.pose.model_scale_fitting import (
    _summarize_scale_groups,
    _summarize_scale_samples,
)


def test_grouped_scale_summaries_match_scalar():
    samples = {
        "a": [1.0, 2.0, 3.0],
        "empty": [],
        "b": [3.0, 8.0, 9.0],
        "c": [4.0, 7.0],
        "d": [5.0],
    }
    expected = {
        name: _summarize_scale_samples(segment_name=name, samples=values)
        for name, values in samples.items()
        if values
    }
    actual = _summarize_scale_groups(scale_samples=samples)
    assert actual.keys() == expected.keys()
    for name, reading in actual.items():
        original = expected[name]
        assert (
            reading.segment_name,
            reading.median_scale,
            reading.relative_dispersion,
            reading.sample_count,
        ) == (
            original.segment_name,
            original.median_scale,
            original.relative_dispersion,
            original.sample_count,
        )
    assert _summarize_scale_groups(scale_samples={}) == {}


@pytest.mark.parametrize("samples", [[0.0], [-1.0], [float("nan")]])
def test_bad_scale_summary_still_raises(samples):
    with pytest.raises(ValueError):
        _summarize_scale_groups(scale_samples={"bad": samples})
