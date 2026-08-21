"""Anatomical segment-length measurement and diagnostics.

Measure body-segment lengths from a canonical-named 3D keypoint time series
and assess whether the data is *human-shaped*: segments in anthropometric
proportion, rigid over time, and left/right symmetric.

The reference is the canonical body model's ``bone_length_ratios`` (each
bone's length as a fraction of standing height — Winter 2009 / Drillis &
Contini 1966). Because the checks divide measured length by the canonical
ratio, they are **height-independent**: a genuinely human skeleton implies
one consistent standing height across every segment, so the spread of
per-segment implied heights is the core "is this human-shaped?" signal.

This module is pure measurement + assessment over a ``{landmark_name:
(frames, 3)}`` array dict. It does not depend on any particular tracker —
callers pass canonical-named positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------
#
# TODO: Replace this hardcoded list with segment definitions derived from
# the ``StandardHuman`` model (SF-SH-5). The canonical model's bone list
# should be the single source of truth for which segments are measured
# and their proximal/distal landmarks.


@dataclass(frozen=True, slots=True)
class SegmentDef:
    """Definition of one limb segment for length measurement."""

    name: str
    proximal: str
    distal: str
    pair: str  # base name shared by left/right pair, e.g. "upper_arm"

    @property
    def bone_key(self) -> str:
        """Key into bone_length_ratios ("parent->child")."""
        return f"{self.proximal}->{self.distal}"


LIMB_SEGMENTS: tuple[SegmentDef, ...] = (
    SegmentDef("left_upper_arm", "left_shoulder", "left_elbow", "upper_arm"),
    SegmentDef("right_upper_arm", "right_shoulder", "right_elbow", "upper_arm"),
    SegmentDef("left_forearm", "left_elbow", "left_wrist", "forearm"),
    SegmentDef("right_forearm", "right_elbow", "right_wrist", "forearm"),
    SegmentDef("left_thigh", "left_hip", "left_knee", "thigh"),
    SegmentDef("right_thigh", "right_hip", "right_knee", "thigh"),
    SegmentDef("left_shank", "left_knee", "left_ankle", "shank"),
    SegmentDef("right_shank", "right_knee", "right_ankle", "shank"),
)


@lru_cache(maxsize=1)
def canonical_bone_length_ratios() -> dict[str, float]:
    """Bone-length ratios (length / standing height) from the canonical body model.

    Reads ``canonical_body.yaml`` directly — avoids the old
    ``AnatomicalStructure`` import chain (which is being replaced by
    ``StandardHuman`` in SF-SH-5). Cached — loaded once per process.

    TODO (SF-SH-5): derive from ``StandardHuman.bones`` instead of
    reading the YAML directly.
    """
    import yaml
    from pathlib import Path

    yaml_path = Path(__file__).parents[1] / "skellymodels" / "tracker_info" / "canonical_body.yaml"  # kinematics/../skellymodels/...
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    ratios = data.get("aspects", {}).get("body", {}).get("bone_length_ratios", {})
    if not ratios:
        raise ValueError(
            "Canonical body model exposes no bone_length_ratios"
        )
    return dict(ratios)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HumanShapeThresholds:
    """Pass/fail bars for human-shape and equivalence checks (balanced)."""

    max_temporal_cv: float = 0.15
    max_proportion_cv: float = 0.15
    max_symmetry_diff: float = 0.15
    min_height_mm: float = 1000.0
    max_height_mm: float = 2200.0
    max_equivalence_diff: float = 0.25
    min_valid_fraction: float = 0.25
    min_assessable_segments: int = 4


DEFAULT_THRESHOLDS = HumanShapeThresholds()


# ---------------------------------------------------------------------------
# Per-segment statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentStats:
    """Summary statistics for one segment over a time series."""

    name: str
    pair: str
    ratio: float
    mean_mm: float
    median_mm: float
    std_mm: float
    mad_mm: float  # median absolute deviation (robust spread)
    min_mm: float
    max_mm: float
    n_valid: int
    n_frames: int

    @property
    def temporal_cv(self) -> float:
        """Robust CV: MAD × 1.4826 / median.

        Robust to transient bad frames (occlusion, foreshortening) that
        would inflate a plain standard deviation.
        """
        robust_std = 1.4826 * self.mad_mm
        return robust_std / self.median_mm if self.median_mm > 0 else float("inf")

    @property
    def implied_height_mm(self) -> float:
        """Standing height implied by this segment's median length."""
        return self.median_mm / self.ratio if self.ratio > 0 else float("nan")

    @property
    def valid_fraction(self) -> float:
        """Fraction of frames with a valid (finite, positive) length."""
        return self.n_valid / self.n_frames if self.n_frames else 0.0

    @property
    def range_mm(self) -> float:
        return self.max_mm - self.min_mm


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_segment_lengths(
    canonical_positions: dict[str, np.ndarray],
    segments: tuple[SegmentDef, ...] = LIMB_SEGMENTS,
) -> dict[str, np.ndarray]:
    """Per-frame Euclidean length of each segment.

    Parameters
    ----------
    canonical_positions : dict[str, (frames, 3) ndarray]
        Canonical-named 3D positions. Missing landmarks or NaN coordinates
        yield NaN lengths for the affected frames.
    segments : tuple of SegmentDef

    Returns
    -------
    dict[str, (frames,) ndarray]
        Per-segment length series. Segments whose endpoints are both
        present in the dict are included.
    """
    lengths: dict[str, np.ndarray] = {}
    for seg in segments:
        proximal = canonical_positions.get(seg.proximal)
        distal = canonical_positions.get(seg.distal)
        if proximal is None or distal is None:
            continue
        proximal = np.asarray(proximal, dtype=float)
        distal = np.asarray(distal, dtype=float)
        lengths[seg.name] = np.linalg.norm(distal - proximal, axis=-1)
    return lengths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SegmentLengthReport:
    """Per-segment stats + aggregate human-shape metrics."""

    stats: dict[str, SegmentStats]
    thresholds: HumanShapeThresholds = DEFAULT_THRESHOLDS

    def assessable(self) -> dict[str, SegmentStats]:
        """Segments with enough valid frames to trust."""
        return {
            name: s
            for name, s in self.stats.items()
            if s.valid_fraction >= self.thresholds.min_valid_fraction
            and s.median_mm > 0
        }

    @property
    def implied_heights(self) -> dict[str, float]:
        return {
            n: s.implied_height_mm for n, s in self.assessable().items()
        }

    @property
    def implied_height_median_mm(self) -> float:
        vals = [
            h for h in self.implied_heights.values() if np.isfinite(h)
        ]
        return float(np.median(vals)) if vals else float("nan")

    @property
    def implied_height_cv(self) -> float:
        vals = [
            h for h in self.implied_heights.values() if np.isfinite(h)
        ]
        if len(vals) < 2:
            return float("inf")
        med = float(np.median(vals))
        return float(np.std(vals)) / med if med > 0 else float("inf")

    def symmetry_diffs(self) -> dict[str, float]:
        """Relative left/right length difference per pair."""
        assessable = self.assessable()
        by_pair: dict[str, dict[str, float]] = {}
        for name, s in assessable.items():
            side = "left" if name.startswith("left_") else "right"
            by_pair.setdefault(s.pair, {})[side] = s.median_mm
        diffs: dict[str, float] = {}
        for pair, sides in by_pair.items():
            if "left" in sides and "right" in sides:
                lo, hi = sides["left"], sides["right"]
                mean = (lo + hi) / 2.0
                diffs[pair] = (
                    abs(lo - hi) / mean if mean > 0 else float("inf")
                )
        return diffs

    def human_shape_violations(
        self, *, check_rigidity: bool = True
    ) -> list[str]:
        """Return human-readable violation strings; empty means pass."""
        t = self.thresholds
        violations: list[str] = []

        assessable = self.assessable()
        if len(assessable) < t.min_assessable_segments:
            violations.append(
                f"only {len(assessable)} segment(s) have "
                f">={t.min_valid_fraction:.0%} valid frames "
                f"(need {t.min_assessable_segments})"
            )
            return violations

        cv = self.implied_height_cv
        if cv > t.max_proportion_cv:
            violations.append(
                f"implied-height CV {cv:.3f} > {t.max_proportion_cv} "
                f"(segments disagree on body scale)"
            )

        median_h = self.implied_height_median_mm
        if not (t.min_height_mm <= median_h <= t.max_height_mm):
            violations.append(
                f"implied standing height {median_h:.0f}mm outside "
                f"[{t.min_height_mm:.0f}, {t.max_height_mm:.0f}]mm"
            )

        if check_rigidity:
            for name, s in assessable.items():
                if s.temporal_cv > t.max_temporal_cv:
                    violations.append(
                        f"{name} temporal CV {s.temporal_cv:.3f} > "
                        f"{t.max_temporal_cv}"
                    )

        for pair, diff in self.symmetry_diffs().items():
            if diff > t.max_symmetry_diff:
                violations.append(
                    f"{pair} left/right differ by {diff:.1%} > "
                    f"{t.max_symmetry_diff:.0%}"
                )

        return violations

    def summary(self) -> str:
        lines = [
            "Segment-length report "
            "(median mm | temporal CV | implied height mm):"
        ]
        for name in sorted(self.stats):
            s = self.stats[name]
            lines.append(
                f"  {name:18s} {s.median_mm:7.1f} | "
                f"cv={s.temporal_cv:5.3f} | H={s.implied_height_mm:7.1f} "
                f"| valid={s.valid_fraction:.0%}"
            )
        lines.append(
            f"  -> implied height: median={self.implied_height_median_mm:.0f}mm "
            f"cv={self.implied_height_cv:.3f}"
        )
        sym = self.symmetry_diffs()
        if sym:
            lines.append(
                "  -> symmetry: "
                + ", ".join(
                    f"{p}={d:.1%}" for p, d in sorted(sym.items())
                )
            )
        return "\n".join(lines)


def report_from_segment_lengths(
    lengths_by_segment: dict[str, np.ndarray],
    *,
    ratios: dict[str, float] | None = None,
    segments: tuple[SegmentDef, ...] = LIMB_SEGMENTS,
    thresholds: HumanShapeThresholds = DEFAULT_THRESHOLDS,
) -> SegmentLengthReport:
    """Build a stats report from already-measured per-segment length series."""
    if ratios is None:
        ratios = canonical_bone_length_ratios()
    by_name = {seg.name: seg for seg in segments}

    stats: dict[str, SegmentStats] = {}
    for name, length_series in lengths_by_segment.items():
        seg = by_name.get(name)
        if seg is None:
            continue
        ratio = ratios.get(seg.bone_key)
        if ratio is None or ratio <= 0.0:
            raise ValueError(
                f"No positive canonical bone-length ratio for "
                f"'{seg.bone_key}'"
            )
        series = np.asarray(length_series, dtype=float)
        finite = series[np.isfinite(series) & (series > 0.0)]
        n_frames = int(series.shape[0])
        if finite.size == 0:
            stats[name] = SegmentStats(
                name, seg.pair, ratio, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0, n_frames,
            )
            continue
        median = float(np.median(finite))
        stats[name] = SegmentStats(
            name=name,
            pair=seg.pair,
            ratio=ratio,
            mean_mm=float(np.mean(finite)),
            median_mm=median,
            std_mm=float(np.std(finite)),
            mad_mm=float(np.median(np.abs(finite - median))),
            min_mm=float(np.min(finite)),
            max_mm=float(np.max(finite)),
            n_valid=int(finite.size),
            n_frames=n_frames,
        )
    return SegmentLengthReport(stats=stats, thresholds=thresholds)


def build_segment_length_report(
    canonical_positions: dict[str, np.ndarray],
    *,
    ratios: dict[str, float] | None = None,
    segments: tuple[SegmentDef, ...] = LIMB_SEGMENTS,
    thresholds: HumanShapeThresholds = DEFAULT_THRESHOLDS,
) -> SegmentLengthReport:
    """Measure segment lengths and build a stats report."""
    return report_from_segment_lengths(
        measure_segment_lengths(canonical_positions, segments),
        ratios=ratios,
        segments=segments,
        thresholds=thresholds,
    )


def equivalence_violations(
    report_a: SegmentLengthReport,
    report_b: SegmentLengthReport,
    *,
    thresholds: HumanShapeThresholds = DEFAULT_THRESHOLDS,
    label_a: str = "A",
    label_b: str = "B",
) -> list[str]:
    """Compare per-segment median lengths between two reports.

    ``report_b`` is the reference (denominator). Returns human-readable
    violations; empty list means equivalent within ``max_equivalence_diff``.
    """
    a_ok = report_a.assessable()
    b_ok = report_b.assessable()
    common = sorted(set(a_ok) & set(b_ok))
    violations: list[str] = []
    if len(common) < thresholds.min_assessable_segments:
        violations.append(
            f"only {len(common)} comparable segment(s) "
            f"(need {thresholds.min_assessable_segments})"
        )
        return violations
    for name in common:
        a = a_ok[name].median_mm
        b = b_ok[name].median_mm
        diff = abs(a - b) / b if b > 0 else float("inf")
        if diff > thresholds.max_equivalence_diff:
            violations.append(
                f"{name}: {label_a}={a:.0f}mm vs {label_b}={b:.0f}mm "
                f"({diff:.1%} > {thresholds.max_equivalence_diff:.0%})"
            )
    return violations
