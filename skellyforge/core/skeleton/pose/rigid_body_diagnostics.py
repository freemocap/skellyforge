"""Segment measurements and residual statistics in the pose's length units."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose


@dataclass(frozen=True, slots=True)
class RigidBodyResidual:
    measured_length: float | None
    reference_length: float | None
    residual: float | None
    reference_kind: str


@dataclass(frozen=True, slots=True)
class ResidualSummary:
    sample_count: int
    unavailable_count: int
    mean: float | None
    rms: float | None
    standard_deviation: float | None


@dataclass(slots=True)
class ResidualAccumulator:
    """Online population statistics; unavailable measurements are counted separately."""

    sample_count: int = 0
    unavailable_count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def add(self, *, residual: float | None) -> None:
        if residual is None:
            self.unavailable_count += 1
            return
        if not math.isfinite(residual):
            raise ValueError('Residual must be finite or unavailable')
        self.sample_count += 1
        delta = residual - self._mean
        self._mean += delta / self.sample_count
        self._m2 += delta * (residual - self._mean)

    def summary(self) -> ResidualSummary:
        if not self.sample_count:
            return ResidualSummary(0, self.unavailable_count, None, None, None)
        variance = max(0.0, self._m2 / self.sample_count)
        return ResidualSummary(self.sample_count, self.unavailable_count, self._mean,
                               math.sqrt(variance + self._mean ** 2), math.sqrt(variance))


def measure_rigid_body_residuals(
    *, skeleton: SkeletonDefinition, pose: SkeletonPose,
    measured_segment_names: frozenset[str], reference_lengths: Mapping[str, float],
    reference_kind: str,
) -> dict[str, RigidBodyResidual]:
    """Compare observed segment scales with an explicit reference, without fitting it."""
    if not measured_segment_names.issubset(skeleton.segments):
        raise ValueError('Measured segments must belong to the skeleton')
    if not set(reference_lengths).issubset(skeleton.segments):
        raise ValueError('Reference segments must belong to the skeleton')
    result: dict[str, RigidBodyResidual] = {}
    for name, segment in skeleton.segments.items():
        observed = pose.segment_poses.get(name)
        measured = observed.scale_estimate * segment.length if observed is not None and name in measured_segment_names else None
        reference = reference_lengths.get(name)
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in (measured, reference)):
            raise ValueError(f'Invalid length for segment {name}')
        result[name] = RigidBodyResidual(
            measured_length=measured, reference_length=reference,
            residual=measured - reference if measured is not None and reference is not None else None,
            reference_kind=reference_kind,
        )
    return result
