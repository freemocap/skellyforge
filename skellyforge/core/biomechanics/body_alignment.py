"""Estimate a frozen body reference from independent anatomical orientation tracks.

Head and torso tracks have equal standing: quality and temporal agreement select the
reference. This estimates a body frame, never a gravity direction or floor height.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import ORTHONORMALITY_TOLERANCE
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SegmentPose
from skellyforge.type_overloads import FloatArray, RigidBodySegmentName


class AlignmentOutcome(StrEnum):
    BODY_REFERENCE = "body_reference"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True, kw_only=True)
class BodyAlignmentConfig:
    minimum_quality: float = 0.5
    minimum_duration_seconds: float = 0.5
    maximum_gap_seconds: float = 0.2
    maximum_deviation_degrees: float = 15.0

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be in (0, 1]")
        if not 0.0 < self.minimum_duration_seconds < float("inf"):
            raise ValueError("minimum_duration_seconds must be finite and positive")
        if not 0.0 < self.maximum_gap_seconds < float("inf"):
            raise ValueError("maximum_gap_seconds must be finite and positive")
        if not 0.0 < self.maximum_deviation_degrees < 90.0:
            raise ValueError("maximum_deviation_degrees must be in (0, 90)")


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BodyReferenceTrack:
    """One anatomical region of ONE subject, with explicit body-axis semantics.

    Rotations map canonical +X right, +Y forward, +Z up into the input world frame.
    They must account for the segment's authored rest orientation; raw segment-local
    rotations are not necessarily body-frame rotations. Quality is reconstruction
    quality in [0,1], not landmark count or an uncalibrated detector score. Missing
    samples have quality zero; arbitrary/unobserved roll must not receive positive quality.
    Origins and all other spatial inputs use the caller's single declared length unit.
    """

    segment_name: RigidBodySegmentName
    timestamps_seconds: FloatArray
    world_from_body: FloatArray
    origins: FloatArray
    quality: FloatArray

    @classmethod
    def from_segment_poses(
        cls,
        *,
        segment_name: RigidBodySegmentName,
        poses: tuple[SegmentPose | None, ...],
        rest_pose: RestPose,
        timestamps_seconds: FloatArray,
        quality: FloatArray,
    ) -> BodyReferenceTrack:
        """Convert measured segment orientations to canonical body-reference axes.

        The relative rotation is observed-world-from-segment times the inverse of
        rest-world-from-segment. Direction-only and transported-roll solutions do not
        measure a full orientation and contribute no alignment evidence.
        """
        if len(poses) != len(timestamps_seconds) or quality.shape != (len(poses),):
            raise ValueError("Pose, timestamp and quality counts must match")
        if not np.all(np.isfinite(quality)) or np.any((quality < 0) | (quality > 1)):
            raise ValueError("Pose quality must be finite and in [0,1]")
        rest_inverse = (
            rest_pose.segment_orientations[segment_name].to_rotation_matrix().T
        )
        rotations = np.full((len(poses), 3, 3), np.nan)
        origins = np.full((len(poses), 3), np.nan)
        evidence_quality = np.zeros(len(poses))
        for index, pose in enumerate(poses):
            if pose is None:
                continue
            if pose.segment_name != segment_name:
                raise ValueError("Body reference track cannot mix segment identities")
            if pose.solved_by is not PoseSolution.RIGID_FIT:
                continue
            rotations[index] = pose.orientation.to_rotation_matrix() @ rest_inverse
            origins[index] = pose.origin.array
            evidence_quality[index] = quality[index]
        return cls(
            segment_name=segment_name,
            timestamps_seconds=timestamps_seconds.copy(),
            world_from_body=rotations,
            origins=origins,
            quality=evidence_quality,
        )

    def __post_init__(self) -> None:
        times = self.timestamps_seconds
        if (
            times.ndim != 1
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "timestamps_seconds must be finite and strictly increasing"
            )
        count = len(times)
        if self.world_from_body.shape != (count, 3, 3) or self.origins.shape != (
            count,
            3,
        ):
            raise ValueError("Body reference rotations/origins must match timestamps")
        if self.quality.shape != (count,) or not np.all(np.isfinite(self.quality)):
            raise ValueError("quality must be a finite vector matching timestamps")
        if np.any((self.quality < 0) | (self.quality > 1)):
            raise ValueError("quality must be in [0,1]")
        valid = self.quality > 0
        rotations = self.world_from_body[valid]
        if not np.all(np.isfinite(rotations)) or not np.all(
            np.isfinite(self.origins[valid])
        ):
            raise ValueError("Positive-quality body samples must have finite geometry")
        if not np.allclose(
            np.swapaxes(rotations, -1, -2) @ rotations,
            np.eye(3),
            atol=ORTHONORMALITY_TOLERANCE,
            rtol=0,
        ) or not np.allclose(
            np.linalg.det(rotations), 1, atol=ORTHONORMALITY_TOLERANCE, rtol=0
        ):
            raise ValueError(
                "Body reference rotations must be proper orthonormal matrices"
            )
        if not self.segment_name:
            raise ValueError("A body reference requires a named anatomical region")


@dataclass(frozen=True, slots=True, kw_only=True)
class BodyAlignmentResult:
    outcome: AlignmentOutcome
    transform: Transform | None
    anchor_segment: RigidBodySegmentName | None
    support_seconds: float
    mean_quality: float
    maximum_deviation_degrees: float


def _rotation_mean(*, rotations: FloatArray, weights: FloatArray) -> FloatArray:
    left, _, right = np.linalg.svd(np.einsum("n,nij->ij", weights, rotations))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def _rotation_deviations(*, rotations: FloatArray, reference: FloatArray) -> FloatArray:
    traces = np.einsum("nij,ij->n", rotations, reference)
    return np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))


def estimate_body_alignment(
    *,
    tracks: tuple[BodyReferenceTrack, ...],
    config: BodyAlignmentConfig,
) -> BodyAlignmentResult:
    """Select the best sustained reference without averaging incompatible body regions.

    Each contiguous qualifying interval is scored by mean quality and angular stability.
    Intervals need the configured dwell time. Sample duration, not frame count, weights
    the rotation/anchor estimate. Missing samples and long gaps split intervals. A head
    track can win without hips, feet, or any other anatomical region being available.
    """
    if len({track.segment_name for track in tracks}) != len(tracks):
        raise ValueError("Provide one track per anatomical region of one subject")
    best: BodyAlignmentResult | None = None
    best_score = -1.0
    for track in tracks:
        start = 0
        count = len(track.timestamps_seconds)
        while start < count:
            if track.quality[start] < config.minimum_quality:
                start += 1
                continue
            end = start + 1
            while end < count:
                if track.quality[end] < config.minimum_quality:
                    break
                if (
                    track.timestamps_seconds[end] - track.timestamps_seconds[end - 1]
                    > config.maximum_gap_seconds
                ):
                    break
                deviation = _rotation_deviations(
                    rotations=track.world_from_body[end : end + 1],
                    reference=track.world_from_body[start],
                )[0]
                if deviation > config.maximum_deviation_degrees:
                    break
                end += 1
            times = track.timestamps_seconds[start:end]
            duration = float(times[-1] - times[0])
            if duration >= config.minimum_duration_seconds:
                intervals = np.diff(times)
                time_weights = np.zeros(len(times))
                time_weights[:-1] += intervals / 2
                time_weights[1:] += intervals / 2
                quality = track.quality[start:end]
                weights = time_weights * quality
                weights /= weights.sum()
                rotations = track.world_from_body[start:end]
                reference = _rotation_mean(rotations=rotations, weights=weights)
                deviations = _rotation_deviations(
                    rotations=rotations, reference=reference
                )
                maximum_deviation = float(np.max(deviations))
                mean_quality = float(np.dot(time_weights, quality) / duration)
                score = mean_quality * (
                    1 - maximum_deviation / config.maximum_deviation_degrees
                )
                if (
                    maximum_deviation <= config.maximum_deviation_degrees
                    and score > best_score
                ):
                    origin = np.einsum("n,ni->i", weights, track.origins[start:end])
                    rotation = reference.T
                    best = BodyAlignmentResult(
                        outcome=AlignmentOutcome.BODY_REFERENCE,
                        transform=Transform(
                            rotation=RotationQuaternion.from_rotation_matrix(
                                matrix=rotation
                            ),
                            translation=Displacement.from_prevalidated_array(
                                array=-rotation @ origin
                            ),
                        ),
                        anchor_segment=track.segment_name,
                        support_seconds=duration,
                        mean_quality=mean_quality,
                        maximum_deviation_degrees=maximum_deviation,
                    )
                    best_score = score
            start = end
    if best is not None:
        return best
    return BodyAlignmentResult(
        outcome=AlignmentOutcome.INSUFFICIENT_EVIDENCE,
        transform=None,
        anchor_segment=None,
        support_seconds=0.0,
        mean_quality=0.0,
        maximum_deviation_degrees=0.0,
    )
