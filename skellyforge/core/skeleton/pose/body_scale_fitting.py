"""Fitting the dimensionless template to a real body: one height, one scale per segment.

The authored template has no size. A landmark's local position is a fraction of body
height, so every segment carries a PROPORTION `p` (`RigidBodySegment.length`) rather than a
length, and the same skeleton describes a toddler and a basketball player. Turning it into
a body means answering one question - how big is this subject - which is a problem precisely
because a camera rarely sees all of a person at once. Sitting at a desk, the legs are simply
not there.

The way out is that `p` makes every visible segment an answer to the SAME question. A
segment observed to be `d` long reports

    s = d / p                   world units per unit body height

and that is the subject's height, read off that one bone. A femur says it, a forearm says
it, the skull says it. So the fit is not "measure the lengths, then guess a height" - it is
one scale field over the skeleton, `s̃`, where each segment either has its own reading or
inherits the pooled one:

    Ĥ   = a robust aggregate of the readings from segments that actually measure something
    s̃ₛ  = that segment's own reading, shrunk toward Ĥ by how much evidence it has
    Lₛ  = pₛ · s̃ₛ

A segment nobody can see has no reading, so `s̃ₛ` IS `Ĥ` and its length is `pₛ · Ĥ`. That is
not a fallback - it is the answer the proportional template exists to give. Your feet under
the desk are as long as your humerus says they are.

## What is allowed to vote

Not every segment measures the subject. A tracker mapping synthesizes some landmarks as
ratios of a measured span - the sternoclavicular joints, the xiphoid process - so a segment
between two of them reports `span × (authored ratio) / p`, which is the template quoting
itself back. Worse, such a segment is nearly NOISE-FREE, so any weighting that rewards
consistency would rank it as the best evidence available. `body_scale_voting_segment_names`
is how a caller says which landmarks are real measurements; only segments built entirely
from those set `Ĥ`. Everything else consumes it.

Non-voting segments still keep their own measured `s̃ₛ`, because their length is where their
endpoints actually are - a bone rendered at `pₛ · Ĥ` when its own landmarks say otherwise
would not reach its own joints.

## What makes it robust

Medians throughout, never means: one badly triangulated frame must not stretch a bone, and
one badly tracked limb must not resize the body. Weights are derived, not dialled - a
segment's vote is worth `p²` (a fixed absolute error on `d` is a relative error on `s` that
grows as `1/p`, so short segments genuinely know less), times how many samples it has, over
how much those samples disagree. Left and right pool their samples for `Ĥ`, because a
one-sided occlusion is common and stature is not sided - but they keep separate lengths, so
a real limb-length difference still shows.

Nothing here measures a floor, assumes a subject is standing, or needs a calibration pose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass

import numpy as np

from skellyforge.core.skeleton.loading.sided_expansion import unsided_name_of
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

DEFAULT_SCALE_WINDOW_FRAMES: int = 30
"""How many of a segment's most recent readings the median is taken over.

At 30 fps this is a one-second memory: long enough that a handful of bad frames cannot move
the median, short enough that the fit follows a subject who walks closer to the cameras (and
so gets triangulated better) within a second."""

DISPERSION_HALF_TRUST: float = 0.05
"""The relative spread at which a segment's readings are worth half as much.

A segment whose readings scatter by 5% of their own median is at the edge of what real
anatomy plus honest triangulation noise produces; beyond that the segment is being
mistracked, not measured. Used both to weight votes for the height and to decide how far a
segment's own length is pulled toward the pooled one."""

SHRINKAGE_PRIOR_SAMPLES: float = 8.0
"""How many clean readings it takes before a segment half-trusts itself over the pooled fit.

This is the strength of the template as a prior, in units of evidence. With zero readings a
segment is entirely the template; at `SHRINKAGE_PRIOR_SAMPLES` clean readings it is half its
own measurement; past that it is mostly itself. Noisy readings count for less - the
threshold is scaled up by the segment's dispersion."""


class InsufficientScaleEvidence(ValueError):
    """No segment has measured the subject, so the body has no size yet.

    Distinct from a bad fit: this is the honest state of a fit that has seen nobody, or has
    seen only landmarks its caller declared synthetic. A caller asks `has_body_scale` before
    asking for the fit rather than being handed a plausible-looking default.
    """


@dataclass(frozen=True, slots=True, eq=False)
class SegmentScaleReading:
    """What one segment's window of readings says, robustly summarized.

    Attributes:
        segment_name: the segment these readings came from.
        median_scale: the median reading, in world units per unit body height.
        relative_dispersion: the median absolute deviation over the median - how much the
            readings disagree, as a fraction of their own size. Zero for a single reading.
        sample_count: how many readings the summary is over.
    """

    segment_name: RigidBodySegmentName
    median_scale: float
    relative_dispersion: float
    sample_count: int


@dataclass(frozen=True, slots=True, eq=False)
class BodyScaleFit:
    """A fitted body: one height, and the scale and length of every segment.

    Every segment of the skeleton appears in `segment_scales` and `segment_lengths`,
    including the ones nothing has seen - that is the point of a proportional template. Use
    `measured_segment_names` to tell which are carrying their own measurement and which are
    wearing the pooled one.

    Attributes:
        body_height: the subject's height, in whatever units the observations were in
            (millimetres, for anything coming through freemocap). Numerically this is the
            scale itself, because the template's height is 1.
        segment_scales: each segment's fitted scale - its own reading shrunk toward
            `body_height`, or `body_height` exactly when it has no reading.
        segment_lengths: each segment's fitted length, `proportion x scale`, in the same
            units as `body_height`.
        measured_segment_names: the segments that contributed a reading of their own.
        voting_segment_names: the subset of those that were allowed to set `body_height`.
    """

    body_height: float
    segment_scales: Mapping[RigidBodySegmentName, float]
    segment_lengths: Mapping[RigidBodySegmentName, float]
    measured_segment_names: frozenset[RigidBodySegmentName]
    voting_segment_names: frozenset[RigidBodySegmentName]

    def __post_init__(self) -> None:
        if not self.body_height > 0.0:
            raise ValueError(
                f"body_height must be positive - got {self.body_height!r}"
            )


def body_scale_voting_segment_names(
    *,
    skeleton: SkeletonDefinition,
    measured_landmark_names: Set[LandmarkNameString],
) -> frozenset[RigidBodySegmentName]:
    """Which segments are entitled to set the body height, given what is really measured.

    A segment votes only when every landmark its scale is read from is a real measurement.
    Which landmarks those are depends on how the segment solves, and that is a static
    property of the segment, so this can be answered once rather than per frame:

    * a segment that rigid-fits reads its scale off ALL of its own landmarks, so all of
      them must be measured. This is deliberately strict - the pelvis is excluded by its
      synthesized iliac crests, and it should be.
    * a segment that solves by direction reads its scale off its origin and its primary, so
      those two must be measured.

    Args:
        skeleton: the skeleton whose segments to judge.
        measured_landmark_names: the landmarks that carry a real measurement. A caller with
            a tracker mapping names the ones it passes through or averages from keypoints,
            and leaves out the ones it synthesizes from authored ratios.

    Returns:
        The names of the segments that may vote. Possibly empty - a caller that declares
        nothing measured gets no voters, and the fit will say so rather than inventing one.
    """
    voting: set[RigidBodySegmentName] = set()
    for name, segment in skeleton.segments.items():
        if segment.supports_rigid_fit:
            required: Iterable[LandmarkNameString] = segment.landmarks.keys()
        else:
            required = (
                segment.frame_definition.origin_point_name,
                segment.frame_definition.primary_point_name,
            )
        if all(landmark in measured_landmark_names for landmark in required):
            voting.add(name)
    return frozenset(voting)


def fit_body_scale(
    *,
    skeleton: SkeletonDefinition,
    scale_samples: Mapping[RigidBodySegmentName, Sequence[float]],
    voting_segment_names: Set[RigidBodySegmentName],
) -> BodyScaleFit:
    """Fit one body height and a per-segment scale field from per-segment scale readings.

    Args:
        skeleton: the skeleton being fitted; supplies each segment's authored proportion.
        scale_samples: per segment, its recent readings of world units per unit body height
            (`SegmentPose.body_scale_estimate`). Segments with no readings may be absent or
            empty; a segment not in the skeleton is an error, not a value to ignore.
        voting_segment_names: which segments may set the height, from
            `body_scale_voting_segment_names`.

    Returns:
        The fit, covering every segment of the skeleton.

    Raises:
        InsufficientScaleEvidence: no voting segment has a reading, so there is no size to
            fit. The caller has nothing to publish and should say so.
        KeyError: `scale_samples` names a segment the skeleton does not have.
    """
    unknown = sorted(name for name in scale_samples if name not in skeleton.segments)
    if unknown:
        raise KeyError(
            f"scale_samples names segments that are not in skeleton {skeleton.name!r} - "
            f"{unknown}"
        )

    readings = {
        name: reading
        for name, samples in scale_samples.items()
        if (reading := _summarize_scale_samples(segment_name=name, samples=samples))
        is not None
    }

    body_height = _pooled_body_height(
        skeleton=skeleton,
        readings=readings,
        voting_segment_names=voting_segment_names,
    )

    segment_scales: dict[RigidBodySegmentName, float] = {}
    segment_lengths: dict[RigidBodySegmentName, float] = {}
    for name, segment in skeleton.segments.items():
        reading = readings.get(name)
        if reading is None:
            scale = body_height
        else:
            # The template is a prior worth `prior_weight` clean readings; a segment whose
            # own readings disagree with each other has to bring proportionally more of
            # them before it outweighs it.
            prior_weight = SHRINKAGE_PRIOR_SAMPLES * (
                1.0 + reading.relative_dispersion / DISPERSION_HALF_TRUST
            )
            own_share = reading.sample_count / (reading.sample_count + prior_weight)
            scale = own_share * reading.median_scale + (1.0 - own_share) * body_height
        segment_scales[name] = scale
        segment_lengths[name] = segment.length * scale

    return BodyScaleFit(
        body_height=body_height,
        segment_scales=segment_scales,
        segment_lengths=segment_lengths,
        measured_segment_names=frozenset(readings),
        voting_segment_names=frozenset(readings) & frozenset(voting_segment_names),
    )


def _summarize_scale_samples(
    *, segment_name: RigidBodySegmentName, samples: Sequence[float]
) -> SegmentScaleReading | None:
    """One segment's readings, as a median and how much they disagree.

    Returns `None` for an empty window - a segment nobody has seen has nothing to say, which
    is different from a segment saying something unusable.
    """
    if len(samples) == 0:
        return None
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    if not median > 0.0:
        raise ValueError(
            f"segment {segment_name!r}: median of its scale readings is {median!r}, which "
            "is not a size. Readings come from `SegmentPose.body_scale_estimate`, which is "
            "positive by construction, so this window was filled from somewhere else."
        )
    absolute_deviation = float(np.median(np.abs(values - median)))
    return SegmentScaleReading(
        segment_name=segment_name,
        median_scale=median,
        relative_dispersion=absolute_deviation / median,
        sample_count=len(values),
    )


def _pooled_body_height(
    *,
    skeleton: SkeletonDefinition,
    readings: Mapping[RigidBodySegmentName, SegmentScaleReading],
    voting_segment_names: Set[RigidBodySegmentName],
) -> float:
    """The subject's height: a weighted median over the segments entitled to say.

    Left and right pool into one vote per anatomical segment. Their readings measure the
    same bone, so pooling doubles the evidence and covers the very ordinary case of one side
    being turned away from every camera - and stature, unlike a limb length, is not sided.
    """
    pooled_medians: dict[str, list[float]] = {}
    pooled_counts: dict[str, int] = {}
    pooled_dispersions: dict[str, list[float]] = {}
    pooled_proportions: dict[str, list[float]] = {}
    for name in sorted(voting_segment_names):
        reading = readings.get(name)
        if reading is None:
            continue
        group = unsided_name_of(name=name)
        pooled_medians.setdefault(group, []).append(reading.median_scale)
        pooled_counts[group] = pooled_counts.get(group, 0) + reading.sample_count
        pooled_dispersions.setdefault(group, []).append(reading.relative_dispersion)
        pooled_proportions.setdefault(group, []).append(skeleton.segments[name].length)

    if not pooled_medians:
        raise InsufficientScaleEvidence(
            "no segment entitled to vote has a scale reading, so the subject has no "
            f"measured size. {len(readings)} segment(s) have readings "
            f"({sorted(readings)[:8]}...) and {len(voting_segment_names)} are entitled to "
            "vote, but the two do not overlap - either nobody is in view yet, or every "
            "visible segment is built from landmarks the caller declared synthetic."
        )

    values: list[float] = []
    weights: list[float] = []
    for group, medians in pooled_medians.items():
        proportion = float(np.mean(pooled_proportions[group]))
        dispersion = float(np.max(pooled_dispersions[group]))
        # `proportion**2` is the inverse-variance weight: a fixed absolute error on the
        # observed distance becomes an error on `d / p` that grows as `1 / p`, so a short
        # segment's reading is genuinely and quadratically less informative than a femur's.
        # No hand-written table of which bones to trust - the geometry says.
        weights.append(
            proportion**2
            * pooled_counts[group]
            / (1.0 + dispersion / DISPERSION_HALF_TRUST)
        )
        values.append(float(np.median(medians)))

    return _weighted_median(values=values, weights=weights)


def _weighted_median(*, values: Sequence[float], weights: Sequence[float]) -> float:
    """The value at which half the weight lies below and half above.

    A median rather than a mean because a limb that is being mistracked should not move the
    answer at all, however confidently it is wrong - a weighted mean lets it pull, a
    weighted median lets it vote once.
    """
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    total = float(np.sum(sorted_weights))
    if not total > 0.0:
        raise ValueError(
            f"weights must sum to a positive number - got {total!r} over "
            f"{len(weights)} weights"
        )
    crossing = int(np.searchsorted(np.cumsum(sorted_weights), 0.5 * total))
    return float(sorted_values[min(crossing, len(sorted_values) - 1)])


class StreamingBodyScaleFitter:
    """A rolling body-scale fit over a live pose stream.

    Holds one bounded window of readings per segment and re-fits on demand. The window is
    the ONLY temporal smoothing here: a segment's median over its window is already steady,
    so the height pooled from those medians is steady too, with no second filter to tune or
    to lag.

    A segment that stops being visible keeps its window. That is deliberate - your femur is
    the same length sitting down as standing up, and dropping the measurement when you sit
    would make the height jump for no anatomical reason. `reset` is how a caller says the
    body itself has changed.
    """

    def __init__(
        self,
        *,
        skeleton: SkeletonDefinition,
        voting_segment_names: Set[RigidBodySegmentName],
        window_frames: int = DEFAULT_SCALE_WINDOW_FRAMES,
    ) -> None:
        if window_frames < 1:
            raise ValueError(f"window_frames must be at least 1 - got {window_frames}")
        unknown = sorted(
            name for name in voting_segment_names if name not in skeleton.segments
        )
        if unknown:
            raise KeyError(
                f"voting_segment_names names segments that are not in skeleton "
                f"{skeleton.name!r} - {unknown}"
            )
        self._skeleton = skeleton
        self._voting_segment_names = frozenset(voting_segment_names)
        self._window_frames = window_frames
        # A plain list used as a ring: the fit reads the whole window every frame anyway, so
        # a deque's cheap ends buy nothing, and this avoids reallocating 61 deques on reset.
        self._windows: dict[RigidBodySegmentName, list[float]] = {}
        self._next_slot: dict[RigidBodySegmentName, int] = {}

    @property
    def voting_segment_names(self) -> frozenset[RigidBodySegmentName]:
        """The segments this fitter lets set the body height."""
        return self._voting_segment_names

    @property
    def has_body_scale(self) -> bool:
        """Whether any segment entitled to vote has been seen yet.

        Ask this before `current_fit`. It is the one question a caller with nobody in front
        of the cameras has to answer, and answering it is not the same as being handed a
        plausible default.
        """
        return any(
            self._windows.get(name) for name in self._voting_segment_names
        )

    def observe_pose(self, *, pose: SkeletonPose) -> None:
        """Record every hydrated segment's reading of how big the subject is."""
        for name, segment_pose in pose.segment_poses.items():
            window = self._windows.get(name)
            if window is None:
                window = []
                self._windows[name] = window
                self._next_slot[name] = 0
            if len(window) < self._window_frames:
                window.append(segment_pose.body_scale_estimate)
            else:
                slot = self._next_slot[name]
                window[slot] = segment_pose.body_scale_estimate
                self._next_slot[name] = (slot + 1) % self._window_frames

    def current_fit(self) -> BodyScaleFit:
        """The fit over everything observed so far.

        Raises:
            InsufficientScaleEvidence: no voting segment has been seen. Guard with
                `has_body_scale`.
        """
        return fit_body_scale(
            skeleton=self._skeleton,
            scale_samples=self._windows,
            voting_segment_names=self._voting_segment_names,
        )

    def reset(self) -> None:
        """Forget every reading - a new body, or a new coordinate frame to measure it in."""
        self._windows.clear()
        self._next_slot.clear()
