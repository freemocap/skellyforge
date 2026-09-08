"""Fit a foot-support plane from sustained stationary contact observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from math import comb

import numpy as np

from skellyforge.core.biomechanics.body_alignment import BodyAlignmentResult
from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.type_overloads import FloatArray, LandmarkNameString


class GroundAlignmentOutcome(StrEnum):
    FOOT_SUPPORT = "foot_support"
    INSUFFICIENT_CONTACTS = "insufficient_contacts"
    AMBIGUOUS_SUPPORT = "ambiguous_support"


@dataclass(frozen=True, slots=True, kw_only=True)
class GroundAlignmentConfig:
    """Distance thresholds use the same length unit as contact positions."""

    maximum_speed: float
    maximum_plane_distance: float
    minimum_spread: float
    minimum_quality: float = 0.5
    minimum_dwell_seconds: float = 0.25
    maximum_gap_seconds: float = 0.2
    minimum_support_fraction: float = 0.75
    maximum_body_tilt_degrees: float = 45.0
    maximum_hypotheses: int = 512

    def __post_init__(self) -> None:
        for value in (
            self.maximum_speed,
            self.maximum_plane_distance,
            self.minimum_spread,
            self.minimum_dwell_seconds,
            self.maximum_gap_seconds,
        ):
            if not 0 < value < float("inf"):
                raise ValueError(
                    "Ground alignment distance/time thresholds must be finite and positive"
                )
        if (
            not 0 < self.minimum_quality <= 1
            or not 0.5 < self.minimum_support_fraction <= 1
        ):
            raise ValueError("Invalid ground alignment quality/support threshold")
        if not 0 < self.maximum_body_tilt_degrees < 90:
            raise ValueError("maximum_body_tilt_degrees must be in (0,90)")
        if self.maximum_hypotheses < 1:
            raise ValueError("maximum_hypotheses must be positive")


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FootContactTrack:
    landmark_name: LandmarkNameString
    timestamps_seconds: FloatArray
    positions: FloatArray
    quality: FloatArray

    def __post_init__(self) -> None:
        if self.timestamps_seconds.ndim != 1 or not np.all(
            np.isfinite(self.timestamps_seconds)
        ):
            raise ValueError("Contact timestamps must be a finite vector")
        count = len(self.timestamps_seconds)
        if np.any(np.diff(self.timestamps_seconds) <= 0):
            raise ValueError("Contact timestamps must increase strictly")
        if self.positions.shape != (count, 3) or self.quality.shape != (count,):
            raise ValueError("Contact positions/quality must match timestamps")
        if not np.all(np.isfinite(self.quality)) or np.any(
            (self.quality < 0) | (self.quality > 1)
        ):
            raise ValueError("Contact quality must be finite and in [0,1]")
        if not np.all(np.isfinite(self.positions[self.quality > 0])):
            raise ValueError("Positive-quality contacts must have finite positions")
        if not self.landmark_name:
            raise ValueError("Contact requires a landmark name")


@dataclass(frozen=True, slots=True, kw_only=True)
class GroundAlignmentResult:
    outcome: GroundAlignmentOutcome
    transform: Transform | None
    contact_episodes: int
    support_fraction: float
    residual_distance: float | None


def _contact_centers(
    *, track: FootContactTrack, config: GroundAlignmentConfig
) -> list[FloatArray]:
    centers: list[FloatArray] = []
    start = 0
    count = len(track.timestamps_seconds)
    while start < count:
        if track.quality[start] < config.minimum_quality:
            start += 1
            continue
        end = start + 1
        while end < count:
            delta = track.timestamps_seconds[end] - track.timestamps_seconds[end - 1]
            if (
                track.quality[end] < config.minimum_quality
                or delta > config.maximum_gap_seconds
            ):
                break
            if (
                np.linalg.norm(track.positions[end] - track.positions[end - 1]) / delta
                > config.maximum_speed
            ):
                break
            end += 1
        if (
            track.timestamps_seconds[end - 1] - track.timestamps_seconds[start]
            >= config.minimum_dwell_seconds
        ):
            centers.append(np.median(track.positions[start:end], axis=0))
        start = end
    return centers


def estimate_ground_alignment(
    *,
    contacts: tuple[FootContactTrack, ...],
    body_reference: BodyAlignmentResult,
    config: GroundAlignmentConfig,
) -> GroundAlignmentResult:
    """Return one scene transform; no scale change and no mutation of input tracks.

    One representative per stationary episode prevents frame count from dominating.
    Each landmark receives equal total weight. Body orientation selects plane sign and
    heading, and rejects grossly incompatible planes; it is not used to fit plane tilt.
    """
    if len({track.landmark_name for track in contacts}) != len(contacts):
        raise ValueError("Provide one contact track per landmark of one subject")
    centers: list[FloatArray] = []
    weights_list: list[float] = []
    for track in contacts:
        episodes = _contact_centers(track=track, config=config)
        centers.extend(episodes)
        if episodes:
            weights_list.extend([1 / len(episodes)] * len(episodes))
    count = len(centers)
    if count < 3 or body_reference.transform is None:
        return GroundAlignmentResult(
            outcome=GroundAlignmentOutcome.INSUFFICIENT_CONTACTS,
            transform=None,
            contact_episodes=count,
            support_fraction=0.0,
            residual_distance=None,
        )
    points = np.asarray(centers)
    weights = np.asarray(weights_list)
    weights /= weights.sum()
    body_basis = body_reference.transform.rotation.to_rotation_matrix().T
    best_support = 0.0
    best_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None
    if comb(count, 3) <= config.maximum_hypotheses:
        hypotheses = list(combinations(range(count), 3))
    else:
        generator = np.random.default_rng(seed=0)
        hypotheses = [
            tuple(
                int(index) for index in generator.choice(count, size=3, replace=False)
            )
            for _ in range(config.maximum_hypotheses)
        ]
    for indices in hypotheses:
        a, b, c = points[list(indices)]
        normal = np.cross(b - a, c - a)
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= MINIMUM_VECTOR_NORM:
            continue
        normal /= magnitude
        if abs(float(normal @ body_basis[:, 2])) < np.cos(
            np.radians(config.maximum_body_tilt_degrees)
        ):
            continue
        mask = np.abs((points - a) @ normal) <= config.maximum_plane_distance
        support = float(weights[mask].sum())
        if support > best_support:
            best_support, best_mask = support, mask
    if best_mask is None or best_support < config.minimum_support_fraction:
        return GroundAlignmentResult(
            outcome=GroundAlignmentOutcome.AMBIGUOUS_SUPPORT,
            transform=None,
            contact_episodes=count,
            support_fraction=best_support,
            residual_distance=None,
        )
    selected = points[best_mask]
    selected_weights = weights[best_mask] / best_support
    center = np.einsum("n,ni->i", selected_weights, selected)
    centered = selected - center
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.einsum("n,ni,nj->ij", selected_weights, centered, centered)
    )
    normal = eigenvectors[:, 0]
    residual = float(np.sqrt(max(0, eigenvalues[0])))
    if (
        eigenvalues[1] < config.minimum_spread**2
        or residual > config.maximum_plane_distance
        or abs(float(normal @ body_basis[:, 2]))
        < np.cos(np.radians(config.maximum_body_tilt_degrees))
    ):
        return GroundAlignmentResult(
            outcome=GroundAlignmentOutcome.AMBIGUOUS_SUPPORT,
            transform=None,
            contact_episodes=count,
            support_fraction=best_support,
            residual_distance=residual,
        )
    if normal @ body_basis[:, 2] < 0:
        normal = -normal
    forward = body_basis[:, 1] - normal * (normal @ body_basis[:, 1])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, normal)
    rotation = np.stack((right, forward, normal))
    return GroundAlignmentResult(
        outcome=GroundAlignmentOutcome.FOOT_SUPPORT,
        transform=Transform(
            rotation=RotationQuaternion.from_rotation_matrix(matrix=rotation),
            translation=Displacement.from_prevalidated_array(array=-rotation @ center),
        ),
        contact_episodes=count,
        support_fraction=best_support,
        residual_distance=residual,
    )
