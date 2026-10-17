"""Interpolate interior gaps, leaving unsupported trajectory ends missing."""
import numpy as np
from dataclasses import dataclass, asdict
from numpy.typing import NDArray


TRAJECTORY_SUPPORT_SECONDS = 0.100
TIME_COMPARISON_TOLERANCE_SECONDS = 1e-12


@dataclass(frozen=True)
class GapFillingReport:
    """Provenance on the unchanged input frame/keypoint grid; JSON-ready via to_dict."""
    algorithm_version: int = 4
    trajectory_support_seconds: float = TRAJECTORY_SUPPORT_SECONDS
    # (keypoint index, start frame index, exclusive stop); indices use the saved grid.
    filled_spans: tuple[tuple[int, int, int], ...] = ()
    discarded_spans: tuple[tuple[int, int, int], ...] = ()
    unsupported_keypoint_indices: tuple[int, ...] = ()
    method: str = "timestamp_linear_visible_intervals"
    # Half-open frame intervals. No interpolation or temporal fitting across blanks.
    active_spans: tuple[tuple[int, int], ...] = ()
    blank_spans: tuple[tuple[int, int], ...] = ()

    def to_dict(self):
        return asdict(self)

    def measured_support(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        support = np.isfinite(points).all(axis=-1)
        for point, start, stop in (*self.filled_spans, *self.discarded_spans):
            if not (0 <= point < support.shape[1] and 0 <= start < stop <= len(support)):
                raise ValueError("Gap-filling provenance does not match the recording grid")
            support[start:stop, point] = False
        return support


    def original_support(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        support = self.measured_support(points)
        for point, start, stop in self.discarded_spans:
            support[start:stop, point] = True
        return support


def _spans(point: int, mask: NDArray[np.bool_]) -> list[tuple[int, int, int]]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(point, int(start), int(stop)) for start, stop in boundaries.reshape(-1, 2)]


def fill_trajectory_gaps(
    *, points: NDArray[np.float64], timestamps_s: NDArray[np.float64],
) -> tuple[NDArray[np.float64], GapFillingReport]:
    """Fill bounded keypoint gaps within visible intervals of one tracked person.

    All-keypoint absence is a hard boundary, including a single blank frame.
    Never extrapolate endpoints or invent an entirely unobserved trajectory.
    Retain the 100 ms support rule from FreeMoCap's version-3 implementation.
    Run before smoothing, hydration and fitting; retain the report so interpolated
    samples cannot vote as measured evidence in scale/alignment estimation.
    """
    points = np.asarray(points, dtype=float)
    timestamps_s = np.asarray(timestamps_s, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3 or len(points) == 0:
        raise ValueError("Expected nonempty (frames, keypoints, 3) trajectories")
    if timestamps_s.shape != (len(points),) or not np.isfinite(timestamps_s).all() or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("Expected finite strictly increasing trajectory timestamps")
    valid = np.isfinite(points).all(axis=-1)
    if np.isinf(points).any() or np.any(np.isfinite(points).any(axis=-1) != valid):
        raise ValueError("Each point must have three finite or three missing coordinates")
    output = points.copy()
    retained = valid.copy()
    visible_spans = _spans(0, valid.any(axis=1))
    spans = []
    unsupported = []
    discarded = []
    for point in range(points.shape[1]):
        available = retained[:, point]
        # Consecutive samples form a run at any frame rate. Missing runs bridge
        # only when their bracketing valid timestamps are at most 100 ms apart.
        for _, start, stop in visible_spans:
            indices = np.flatnonzero(available[start:stop]) + start
            breaks = np.flatnonzero((np.diff(indices) > 1) &
                (np.diff(timestamps_s[indices]) > TRAJECTORY_SUPPORT_SECONDS + TIME_COMPARISON_TOLERANCE_SECONDS)) + 1
            for run in np.split(indices, breaks):
                if len(run) < 2 or timestamps_s[run[-1]] - timestamps_s[run[0]] < TRAJECTORY_SUPPORT_SECONDS - TIME_COMPARISON_TOLERANCE_SECONDS:
                    available[run] = False
        rejected = valid[:, point] & ~available
        discarded.extend(_spans(point, rejected))
        if not available.any():
            unsupported.append(point)
            output[:, point] = np.nan
    output[~retained] = np.nan
    active = tuple((start, stop) for _, start, stop in _spans(0, retained.any(axis=1)))
    for start, stop in active:
        for point in range(points.shape[1]):
            available = retained[start:stop, point]
            if not available.any():
                continue
            times = timestamps_s[start:stop]
            for axis in range(3):
                output[start:stop, point, axis] = np.interp(
                    times, times[available], points[start:stop, point, axis][available],
                    left=np.nan, right=np.nan)
    for point in range(points.shape[1]):
        spans.extend(_spans(point, ~retained[:, point] & np.isfinite(output[:, point]).all(axis=-1)))
    blanks = tuple((start, stop) for _, start, stop in _spans(0, ~retained.any(axis=1)))
    return output, GapFillingReport(filled_spans=tuple(spans), discarded_spans=tuple(discarded),
        unsupported_keypoint_indices=tuple(unsupported), active_spans=active, blank_spans=blanks)
