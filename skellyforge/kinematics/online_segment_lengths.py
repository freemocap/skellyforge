"""Rolling-window median segment-length estimation, realtime and posthoc.

The ONE length estimator both pipelines use: per-segment median of the
origin→distal distance, NaN-excluded. The only difference is the window —
a rolling duration for streaming, ``None`` (unbounded) for posthoc, which is
not degraded to match realtime.

Keyed by **segment name**: length is a property of a segment, so the
``"parent->child"`` arrow key and its ``split("->")`` parsing are gone. A
plain rolling median — no trust region, no agreement gating, no error
weighting, no age decay. The median is inherently robust to the occasional
mis-triangulated frame, and lengths are measured only from really-observed
(non-extrapolated) landmarks, so a hidden limb contributes nothing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentLengthEstimator:
    """Per-segment rolling-window median length estimator.

    ``update`` is called once per frame with the current named landmark
    positions; ``lengths`` returns the current median length estimate (mm)
    per segment.

    Parameters
    ----------
    segment_endpoints : dict[str, tuple[str, str]]
        ``segment_name → (origin, distal)``. Defines which segments are
        tracked, as the two landmarks whose distance is the segment length.
    segment_seeds : dict[str, float]
        ``segment_name → seed length (mm)`` (anthropometric ratio × height) —
        the fallback while a segment's window is empty.
    window_seconds : float | None
        Rolling-window duration in seconds; ``None`` = unbounded (posthoc).
        A measurement is dropped once it is strictly older than
        ``window_seconds`` relative to the most recent ``update`` timestamp;
        with ``None`` nothing is ever evicted — the window is the whole
        recording.
    """

    segment_endpoints: dict[str, tuple[str, str]]
    segment_seeds: dict[str, float]
    window_seconds: float | None

    _windows: dict[str, deque[tuple[float, float]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if set(self.segment_endpoints) != set(self.segment_seeds):
            raise ValueError(
                "segment_endpoints and segment_seeds must name the same segments"
            )
        self._windows = {name: deque() for name in self.segment_endpoints}

    @property
    def endpoints(self) -> dict[str, tuple[str, str]]:
        """``segment_name → (origin, distal)``."""
        return dict(self.segment_endpoints)

    @property
    def seeds(self) -> dict[str, float]:
        """Anthropometric seed (mm) per segment — the empty-window fallback."""
        return dict(self.segment_seeds)

    def update(
        self, positions: dict[str, np.ndarray], *, t: float
    ) -> None:
        """Append this frame's per-segment length measurements, then age windows.

        A segment is measured only when both of its landmarks are present.
        Every window — measured this frame or not — drops samples strictly
        older than ``window_seconds`` so a segment that leaves view eventually
        falls back to its seed. With ``window_seconds=None`` nothing is
        dropped.
        """
        cutoff = None if self.window_seconds is None else t - self.window_seconds
        for segment_name, (origin_kp, distal_kp) in self.segment_endpoints.items():
            window = self._windows[segment_name]
            p = positions.get(origin_kp)
            c = positions.get(distal_kp)
            if p is not None and c is not None:
                length = float(
                    np.linalg.norm(
                        np.asarray(c, dtype=float)
                        - np.asarray(p, dtype=float)
                    )
                )
                if np.isfinite(length) and length > 0.0:
                    window.append((t, length))
            if cutoff is not None:
                while window and window[0][0] < cutoff:
                    window.popleft()

    @property
    def lengths(self) -> dict[str, float]:
        """Current median length estimate (mm) per segment.

        Returns the seed for any segment whose window is empty.
        """
        out: dict[str, float] = {}
        for segment_name, window in self._windows.items():
            if window:
                out[segment_name] = float(
                    np.median([length for _, length in window])
                )
            else:
                out[segment_name] = self.segment_seeds[segment_name]
        return out

    def reset(self) -> None:
        """Forget all measurements — estimates fall back to seeds."""
        for window in self._windows.values():
            window.clear()
