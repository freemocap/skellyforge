"""Fitting a dimensionless skeleton to a real thing: one scale, one scale per segment.

An authored skeleton has no size. A landmark's local position is a fraction of the
skeleton's own REFERENCE UNIT - body height for the standard human, square length for a
charuco board - so every segment carries a PROPORTION `p` (`RigidBodySegment.length`)
rather than a length, and the same skeleton describes a toddler and a basketball player.
Turning it into a thing means answering one question - how big is this one - which is a
problem precisely because a camera rarely sees all of anything at once. Sitting at a desk,
the legs are simply not there.

The way out is that `p` makes every visible segment an answer to the SAME question. A
segment observed to be `d` long reports

    s = d / p                   world units per unit of the model's reference unit

and that is the model's size, read off that one segment. A femur says it, a forearm says
it, the skull says it. So the fit is not "measure the lengths, then guess a size" - it is
one scale field over the skeleton, `s̃`, where each segment either has its own reading or
inherits the pooled one:

    Ŝ   = a robust aggregate of the readings from segments that actually measure something
    s̃ₛ  = that segment's own reading, shrunk toward Ŝ by how much evidence it has
    Lₛ  = pₛ · s̃ₛ

A segment nobody can see has no reading, so `s̃ₛ` IS `Ŝ` and its length is `pₛ · Ŝ`. That is
not a fallback - it is the answer a proportional skeleton exists to give. Your feet under
the desk are as long as your humerus says they are.

What the fitted scale MEANS depends on what the skeleton normalized against: for the human
it is stature in millimetres; for a board authored at `square_length = 1.0` it is the
measured square length, which is directly comparable to the value a user entered at
calibration.

## What is allowed to vote

Not every segment measures the model. A tracker mapping synthesizes some landmarks as
ratios of a measured span - the sternoclavicular joints, the xiphoid process - so a segment
between two of them reports `span × (authored ratio) / p`, which is the template quoting
itself back. Worse, such a segment is nearly NOISE-FREE, so any weighting that rewards
consistency would rank it as the best evidence available. `scale_voting_segment_names`
is how a caller says which landmarks are real measurements; only segments built entirely
from those set `Ŝ`. Everything else consumes it.

Non-voting segments still keep their own measured `s̃ₛ`, because their length is where their
endpoints actually are - a bone rendered at `pₛ · Ŝ` when its own landmarks say otherwise
would not reach its own joints.

## What makes it robust

Medians throughout, never means: one badly triangulated frame must not stretch a bone, and
one badly tracked limb must not resize the body. Weights are derived, not dialled - a
segment's vote is worth `p²` (a fixed absolute error on `d` is a relative error on `s` that
grows as `1/p`, so short segments genuinely know less), times how many samples it has, over
how much those samples disagree. Left and right pool their samples for `Ŝ`, because a
one-sided occlusion is common and stature is not sided - but they keep separate lengths, so
a real limb-length difference still shows.

Nothing here measures a floor, assumes a subject is standing, or needs a calibration pose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
import math

import numpy as np

from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition  # noqa: TC001 - runtime type checking
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose  # noqa: TC001 - runtime type checking
from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor
from skellyforge.core.skeleton.pose.scale_tensor import ScaleTensor
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
    """No segment has measured the model, so it has no size yet.

    Distinct from a bad fit: this is the honest state of a fit that has seen nobody, or has
    seen only landmarks its caller declared synthetic. A caller asks `has_model_scale` before
    asking for the fit rather than being handed a plausible-looking default.
    """


@dataclass(frozen=True, slots=True, eq=False)
class SegmentScaleReading:
    """What one segment's window of readings says, robustly summarized.

    Attributes:
        segment_name: the segment these readings came from.
        median_scale: the median reading, in world units per unit of the model's reference unit.
        relative_dispersion: the median absolute deviation over the median - how much the
            readings disagree, as a fraction of their own size. Zero for a single reading.
        sample_count: how many readings the summary is over.
    """

    segment_name: RigidBodySegmentName
    median_scale: float
    relative_dispersion: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class ModelScaleFit:
    """A fitted body: one height, and the scale and length of every segment.

    Every segment of the skeleton appears in `segment_scales` and `segment_lengths`,
    including the ones nothing has seen - that is the point of a proportional template. Use
    `measured_segment_names` to tell which are carrying their own measurement and which are
    wearing the pooled one.

    Attributes:
        fitted_scale: the model's size, in whatever units the observations were in
            (millimetres, for anything coming through freemocap). Numerically this is the
            scale itself, because the template's height is 1.
        segment_scales: each segment's fitted scale - its own reading shrunk toward
            `fitted_scale`, or `fitted_scale` exactly when it has no reading.
        segment_lengths: each segment's fitted length, `proportion x scale`, in the same
            units as `fitted_scale`.
        measured_segment_names: the segments that contributed a reading of their own.
        voting_segment_names: the subset of those that were allowed to set `fitted_scale`.
    """

    fitted_scale: float
    segment_scales: Mapping[RigidBodySegmentName, float]
    segment_lengths: Mapping[RigidBodySegmentName, float]
    measured_segment_names: frozenset[RigidBodySegmentName]
    voting_segment_names: frozenset[RigidBodySegmentName]

    def __post_init__(self) -> None:
        if not math.isfinite(self.fitted_scale) or not self.fitted_scale > 0.0:
            raise ValueError(
                f"fitted_scale must be positive - got {self.fitted_scale!r}"
            )
        if not self.segment_scales or set(self.segment_scales) != set(self.segment_lengths):
            raise ValueError("Fit scales and lengths must cover the same nonempty segment set")
        if any(not name for name in self.segment_scales):
            raise ValueError("Fit segment names must be nonempty")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.segment_scales.values()):
            raise ValueError("Segment scales must be finite and positive")
        if any(not math.isfinite(value) or value < 0.0 for value in self.segment_lengths.values()):
            raise ValueError("Segment lengths must be finite and nonnegative")
        if not self.measured_segment_names.issubset(self.segment_scales):
            raise ValueError("Measured segments must belong to the fit")
        if not self.voting_segment_names.issubset(self.measured_segment_names):
            raise ValueError("Voting segments must have measured evidence")


def scale_voting_segment_names(
    *,
    skeleton: SkeletonDefinition,
    measured_landmark_names: Set[LandmarkNameString],
) -> frozenset[RigidBodySegmentName]:
    """Which segments are entitled to set the fitted scale, given what is really measured.

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


def fit_model_scale_arrays(*, skeleton: SkeletonDefinition, segment_names: tuple[str, ...],
                          scales: np.ndarray, segment_present: np.ndarray,
                          voting_segment_names: Set[RigidBodySegmentName]) -> ModelScaleFit:
    """Fit full-recording array evidence once; no per-frame streaming-fitter calls.

    Preserve authored segment order and each segment's input sample order.
    Missing values never become scale evidence. This is for a
    whole-recording fit, not a replacement for a live rolling window.
    """
    values = np.asarray(scales, dtype=float)
    present = np.asarray(segment_present)
    if tuple(skeleton.segments) != segment_names:
        raise ValueError("Segment axis must match skeleton order")
    if values.ndim != 2 or values.shape[1] != len(segment_names) or present.dtype != np.bool_ or present.shape != values.shape:
        raise ValueError("Expected scales and boolean presence (frames, segments)")
    if not np.isfinite(values[present]).all() or np.any(values[present] <= 0):
        raise ValueError("Present scale evidence must be finite and positive")
    from skellyforge.core.skeleton.pose.scale_tensor import ScaleGeometry
    return ScaleGeometry(skeleton=skeleton, voting_segment_names=voting_segment_names).fit(
        scales=values, segment_present=present).to_fit()



def fit_model_scale(
    *,
    skeleton: SkeletonDefinition,
    scale_samples: Mapping[RigidBodySegmentName, Sequence[float] | np.ndarray],
    voting_segment_names: Set[RigidBodySegmentName],
) -> ModelScaleFit:
    """Adapt ragged per-segment readings to the shared numerical scale fitter.

    Args:
        skeleton: the skeleton being fitted; supplies each segment's authored proportion.
        scale_samples: per segment, its recent readings of world units per unit of the model's reference unit
            (`SegmentPose.scale_estimate`). Segments with no readings may be absent or
            empty; a segment not in the skeleton is an error, not a value to ignore.
        voting_segment_names: which segments may set the height, from
            `scale_voting_segment_names`.

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

    # Ragged mappings are accepted only at this public boundary. Each column
    # contributes its own observations; unused rows are explicitly absent.
    segment_names = tuple(skeleton.segments)
    row_count = max((len(samples) for samples in scale_samples.values()), default=0)
    values = np.full((row_count, len(segment_names)), np.nan, dtype=np.float64)
    present = np.zeros(values.shape, dtype=bool)
    for column, name in enumerate(segment_names):
        samples = np.asarray(scale_samples.get(name, ()), dtype=np.float64)
        if samples.ndim != 1:
            raise ValueError(f"segment {name!r}: scale readings must have one sample axis")
        if not np.isfinite(samples).all() or np.any(samples <= 0):
            raise ValueError(f"segment {name!r}: scale readings must be finite and positive")
        values[:len(samples), column] = samples
        present[:len(samples), column] = True
    from skellyforge.core.skeleton.pose.scale_tensor import ScaleGeometry
    return ScaleGeometry(skeleton=skeleton, voting_segment_names=voting_segment_names).fit(
        scales=values, segment_present=present).to_fit()



def _summarize_scale_groups(*, scale_samples: Mapping[RigidBodySegmentName, Sequence[float] | np.ndarray]) -> dict[RigidBodySegmentName, SegmentScaleReading]:
    """Batch medians for equally sized windows without padding missing evidence."""
    groups: dict[int, list[str]] = {}
    for name, samples in scale_samples.items():
        if len(samples):
            groups.setdefault(len(samples), []).append(name)
    summaries = {}
    for count, names in groups.items():
        values = np.asarray([scale_samples[name] for name in names], dtype=np.float64)
        medians = np.median(values, axis=1)
        if np.any(~(medians > 0.0)):
            # Keep the original named error and validation semantics.
            for name in names:
                _summarize_scale_samples(segment_name=name, samples=scale_samples[name])
        dispersion = np.median(np.abs(values - medians[:, None]), axis=1) / medians
        for name, middle, spread in zip(names, medians, dispersion, strict=True):
            summaries[name] = SegmentScaleReading(segment_name=name, median_scale=float(middle),
                relative_dispersion=float(spread), sample_count=count)
    return {name: summaries[name] for name in scale_samples if name in summaries}


def _summarize_scale_samples(
    *, segment_name: RigidBodySegmentName, samples: Sequence[float] | np.ndarray
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
            "is not a size. Readings come from `SegmentPose.scale_estimate`, which is "
            "positive by construction, so this window was filled from somewhere else."
        )
    absolute_deviation = float(np.median(np.abs(values - median)))
    return SegmentScaleReading(
        segment_name=segment_name,
        median_scale=median,
        relative_dispersion=absolute_deviation / median,
        sample_count=len(values),
    )


def _weighted_median(*, values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray) -> float:
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


class StreamingModelScaleFitter:
    """A rolling model-scale fit over a live pose stream.

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
        from skellyforge.core.skeleton.pose.scale_tensor import ScaleGeometry
        self._geometry = ScaleGeometry(skeleton=skeleton, voting_segment_names=voting_segment_names)
        self._scale_values = np.full((window_frames, len(self._geometry.segment_names)), np.nan)
        self._sample_counts = np.zeros(len(self._geometry.segment_names), np.int64)
        self._next_slots = np.zeros_like(self._sample_counts)

    @property
    def voting_segment_names(self) -> frozenset[RigidBodySegmentName]:
        """The segments this fitter lets set the fitted scale."""
        return self._voting_segment_names

    @property
    def has_model_scale(self) -> bool:
        """Whether any segment entitled to vote has been seen yet.

        Ask this before `current_fit`. It is the one question a caller with nobody in front
        of the cameras has to answer, and answering it is not the same as being handed a
        plausible default.
        """
        return bool(np.any((self._sample_counts > 0) & self._geometry.voting_mask))

    def observe_pose(self, *, pose: SkeletonPose) -> None:
        """Adapt an existing object caller to the shared dense evidence operation."""
        from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor
        self.observe_tensor(poses=PoseTensor.from_pose(pose=pose, segment_names=self._geometry.segment_names))

    def observe_tensor(self, *, poses: PoseTensor) -> None:
        """Append each segment's observed scales; gaps consume no ring positions.

        A chunk larger than the ring discards only readings superseded within
        that same chunk. Every retained scatter target is unique.
        """
        if poses.segment_names != self._geometry.segment_names:
            raise ValueError("Pose segment identities must match the scale fitter")
        present = poses.segment_present
        additions = present.sum(axis=0)
        ordinals = np.cumsum(present, axis=0) - 1
        retained = present & (ordinals >= (additions - self._window_frames)[None, :])
        samples, columns = np.nonzero(retained)
        slots = (self._next_slots[columns] + ordinals[samples, columns]) % self._window_frames
        self._scale_values[slots, columns] = poses.scale_estimates[samples, columns]
        self._next_slots[:] = (self._next_slots + additions) % self._window_frames
        self._sample_counts[:] = np.minimum(self._window_frames, self._sample_counts + additions)

    def current_tensor(self) -> ScaleTensor:
        """Return the same fit as current_fit with dense named numerical fields."""
        present = np.arange(self._window_frames)[:, None] < self._sample_counts[None, :]
        return self._geometry.fit(scales=self._scale_values, segment_present=present)

    def current_fit(self) -> ModelScaleFit:
        """The fit over everything observed so far.

        Raises:
            InsufficientScaleEvidence: no voting segment has been seen. Guard with
                `has_model_scale`.
        """
        return self.current_tensor().to_fit()

    def reset(self) -> None:
        """Forget every reading - a new body, or a new coordinate frame to measure it in."""
        self._scale_values.fill(np.nan)
        self._sample_counts.fill(0)
        self._next_slots.fill(0)
