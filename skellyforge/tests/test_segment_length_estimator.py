"""The one segment-length estimator: rolling window vs unbounded, seeds."""

import statistics

import numpy as np
import pytest

from skellyforge.kinematics.online_segment_lengths import SegmentLengthEstimator


def _make(**overrides) -> SegmentLengthEstimator:
    kwargs = dict(
        segment_endpoints={
            "upper_arm": ("shoulder", "elbow"),
            "lower_arm": ("elbow", "wrist"),
        },
        segment_seeds={"upper_arm": 100.0, "lower_arm": 80.0},
        window_seconds=10.0,
    )
    kwargs.update(overrides)
    return SegmentLengthEstimator(**kwargs)


def _ab(length: float) -> dict[str, np.ndarray]:
    """Positions making |elbow - shoulder| == length (lower_arm unmeasured)."""
    return {"shoulder": np.array([0.0, 0.0, 0.0]), "elbow": np.array([0.0, length, 0.0])}


def test_lengths_are_seeds_before_any_update():
    estimator = _make()
    assert estimator.lengths == {"upper_arm": 100.0, "lower_arm": 80.0}


def test_single_measurement_used_immediately():
    estimator = _make()
    estimator.update(_ab(150.0), t=0.0)
    assert estimator.lengths["upper_arm"] == 150.0
    assert estimator.lengths["lower_arm"] == 80.0  # never measured -> seed


def test_lengths_is_median_of_window():
    estimator = _make()
    for i, length in enumerate((140.0, 150.0, 160.0)):
        estimator.update(_ab(length), t=float(i))
    assert estimator.lengths["upper_arm"] == 150.0


def test_median_of_even_count_is_mean_of_middle_two():
    estimator = _make()
    for i, length in enumerate((140.0, 160.0)):
        estimator.update(_ab(length), t=float(i))
    assert estimator.lengths["upper_arm"] == 150.0


def test_segment_with_missing_keypoint_is_not_measured():
    estimator = _make()
    estimator.update({"shoulder": np.array([0.0, 0.0, 0.0])}, t=0.0)
    assert estimator.lengths["upper_arm"] == 100.0
    assert estimator.lengths["lower_arm"] == 80.0


def test_rolling_window_drops_samples_older_than_the_window():
    # strictly older is evicted; exactly at the boundary survives
    estimator = _make(window_seconds=10.0)
    estimator.update(_ab(100.0), t=0.0)
    estimator.update(_ab(200.0), t=5.0)
    assert estimator.lengths["upper_arm"] == 150.0  # median {100, 200}
    estimator.update(_ab(300.0), t=11.0)  # t=0 is >10s old -> evicted
    assert estimator.lengths["upper_arm"] == 250.0  # median {200, 300}


def test_unseen_segment_falls_back_to_seed_after_window():
    estimator = _make(window_seconds=10.0)
    estimator.update(_ab(150.0), t=0.0)
    assert estimator.lengths["upper_arm"] == 150.0
    estimator.update(
        {"elbow": np.array([0.0, 0.0, 0.0]), "wrist": np.array([0.0, 90.0, 0.0])},
        t=100.0,
    )
    assert estimator.lengths["upper_arm"] == 100.0  # window emptied -> seed
    assert estimator.lengths["lower_arm"] == 90.0   # freshly measured


def test_reset_restores_seeds():
    estimator = _make()
    estimator.update(_ab(150.0), t=0.0)
    assert estimator.lengths["upper_arm"] == 150.0
    estimator.reset()
    assert estimator.lengths == {"upper_arm": 100.0, "lower_arm": 80.0}


def test_unbounded_window_reproduces_the_batch_median():
    # Posthoc passes an unbounded window: the estimate equals the batch median
    # over the whole recording — no degradation to match realtime. The absurd
    # inter-frame gaps prove nothing is evicted.
    unbounded = _make(window_seconds=None)
    lengths = [110.0, 95.0, 130.0, 120.0, 140.0]
    for i, length in enumerate(lengths):
        unbounded.update(_ab(length), t=float(i) * 1000.0)
    assert unbounded.lengths["upper_arm"] == statistics.median(lengths)


def test_a_segment_with_no_samples_falls_back_to_its_anthropometric_seed():
    estimator = _make()
    estimator.update(_ab(150.0), t=0.0)  # upper_arm measured; lower_arm never
    assert estimator.lengths == {"upper_arm": 150.0, "lower_arm": 80.0}


def test_endpoints_and_seeds_must_name_the_same_segments():
    with pytest.raises(ValueError, match="same segments"):
        SegmentLengthEstimator(
            segment_endpoints={"upper_arm": ("shoulder", "elbow")},
            segment_seeds={"lower_arm": 80.0},
            window_seconds=10.0,
        )
