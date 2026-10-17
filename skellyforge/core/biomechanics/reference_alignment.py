"""Select one reference transform while respecting the caller's existing frame."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from skellyforge.core.biomechanics.body_alignment import (
    BodyAlignmentConfig,
    BodyAlignmentResult,
    BodyReferenceTrack,
    estimate_body_alignment,
)
from skellyforge.core.biomechanics.ground_alignment import (
    FootContactTrack,
    GroundAlignmentConfig,
    GroundAlignmentResult,
    estimate_ground_alignment,
)
from skellyforge.core.math.geometry.transform_math import Transform


class ReferenceAlignmentOutcome(StrEnum):
    PRESERVED_REFERENCE = "preserved_reference"
    DISABLED = "disabled"
    FOOT_SUPPORT = "foot_support"
    BODY_REFERENCE = "body_reference"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        # Older recordings used a ground-specific name for preserving any frame.
        if value == "explicit_ground":
            return cls.PRESERVED_REFERENCE
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceAlignmentRequest:
    enabled: bool
    preserve_reference_frame: bool
    """Keep supplied geometry; this does not assert that a floor was estimated."""
    body_tracks: tuple[BodyReferenceTrack, ...]
    foot_contacts: tuple[FootContactTrack, ...]
    body_config: BodyAlignmentConfig
    ground_config: GroundAlignmentConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceAlignmentResult:
    outcome: ReferenceAlignmentOutcome
    transform: Transform
    body_evidence: BodyAlignmentResult | None
    ground_evidence: GroundAlignmentResult | None


def estimate_reference_alignment(
    *, request: ReferenceAlignmentRequest
) -> ReferenceAlignmentResult:
    """Use foot support, then body reference, otherwise retain the supplied frame.

    An identity transform means no additional alignment; it never reverses an existing
    calibration transform. This function estimates once and does not own a live lifecycle.
    """
    if request.preserve_reference_frame or not request.enabled:
        return ReferenceAlignmentResult(
            outcome=(
                ReferenceAlignmentOutcome.PRESERVED_REFERENCE
                if request.preserve_reference_frame
                else ReferenceAlignmentOutcome.DISABLED
            ),
            transform=Transform.identity(),
            body_evidence=None,
            ground_evidence=None,
        )
    body = estimate_body_alignment(
        tracks=request.body_tracks, config=request.body_config
    )
    ground = estimate_ground_alignment(
        contacts=request.foot_contacts,
        body_reference=body,
        config=request.ground_config,
    )
    if ground.transform is not None:
        outcome = ReferenceAlignmentOutcome.FOOT_SUPPORT
        transform = ground.transform
    elif body.transform is not None:
        outcome = ReferenceAlignmentOutcome.BODY_REFERENCE
        transform = body.transform
    else:
        outcome = ReferenceAlignmentOutcome.INSUFFICIENT_EVIDENCE
        transform = Transform.identity()
    return ReferenceAlignmentResult(
        outcome=outcome,
        transform=transform,
        body_evidence=body,
        ground_evidence=ground,
    )
