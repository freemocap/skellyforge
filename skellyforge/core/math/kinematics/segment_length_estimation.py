"""Per-segment length estimation: a pure (result, state) action.

Estimates each segment length from live hydrated landmarks as the rolling
median of the origin -> distal distance (the primary axis). This is the
subject-adaptive counterpart to the segment derived nominal length: the
nominal rest length is the empty-window seed, and the median over a rolling
window refines it toward the live subject true proportions.

Frozen state flows in and out explicitly (state -> (result, state)), matching
the orientation solver. No mutation, no authored endpoint tables: the
endpoints (origin landmark + primary-axis target landmark) and the seeds (the
segment derived length) come straight from the skeleton objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skellyforge.type_overloads import FloatArray

from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton


@dataclass(frozen=True, slots=True)
class SegmentLengthResult:
    """Per-segment length estimates for one frame (result only)."""

    lengths: dict[str, float]


@dataclass(frozen=True, slots=True)
class SegmentLengthState:
    """The estimator memory across frames (state only)."""

    timestamp_seconds: float = 0.0
    # segment name -> rolling window of (timestamp, length) samples, oldest first
    windows: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SegmentLengthState":
        return cls()


def estimate_segment_lengths(
    skeleton: HumanSkeleton,
    landmarks: dict[str, FloatArray],
    *,
    timestamp_seconds: float,
    window_seconds: float | None,
    state: SegmentLengthState,
) -> tuple[SegmentLengthResult, SegmentLengthState]:
    """Estimate each segment length from this frame hydrated landmarks.

    For every segment, measure the distance between its origin landmark and
    its primary-axis target landmark (both must be present and finite), append
    it to the segment rolling window, drop samples older than window_seconds
    (None = unbounded, the posthoc form), and take the median. A segment with
    an empty window falls back to its derived nominal length.

    Returns (result, state): the result is the per-segment median length (mm);
    the state is the updated rolling windows.
    """
    previous = state
    cutoff = None if window_seconds is None else timestamp_seconds - window_seconds

    new_windows: dict[str, tuple[tuple[float, float], ...]] = {}
    lengths: dict[str, float] = {}

    for segment in skeleton.segments:
        window = previous.windows.get(segment.name, ())
        origin = landmarks.get(segment.origin_landmark.name)
        distal = landmarks.get(segment.primary_axis.target_landmark)
        if origin is not None and distal is not None:
            measured = float(
                np.linalg.norm(
                    np.asarray(distal, dtype=np.float64)
                    - np.asarray(origin, dtype=np.float64)
                )
            )
            if np.isfinite(measured) and measured > 0.0:
                window = window + ((timestamp_seconds, measured),)
        if cutoff is not None:
            window = tuple(sample for sample in window if sample[0] >= cutoff)
        new_windows[segment.name] = window
        if window:
            lengths[segment.name] = float(np.median([sample[1] for sample in window]))
        else:
            lengths[segment.name] = segment.length

    return (
        SegmentLengthResult(lengths=lengths),
        SegmentLengthState(timestamp_seconds=timestamp_seconds, windows=new_windows),
    )
