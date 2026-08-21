"""Segment-based center of mass (de Leva 1996), from the standard-human segment layer.

Whole-body CoM = mass-weighted sum of per-segment CoM positions, each located from
its SEGMENT pose (origin landmark + de Leva com_fraction along the segment own long
axis). This is the segment-based replacement for the deprecated keypoint method; it
consumes the hydrated segment layer, never raw tracker keypoints.

de Leva 8-segment model maps onto the standard-human (VRM-aligned) segment layer:

- 1:1 segments (CoM rides the segment own origin -> distal axis):
    upper_arm -> upper_arm, forearm -> lower_arm, thigh -> upper_leg,
    shank -> lower_leg, foot -> foot (de Leva foot = ankle -> ball; toes massless).
- composite spans (de Leva segment covers several standard-human segments; its CoM
    lies on the span between two segment-layer landmarks):
    head  = head_vertex -> neck_center       (de Leva head includes the neck)
    trunk = neck_center -> hips_center       (T1 ~= de Leva suprasternale)
    hand  = wrist -> hand_middle_finger_tip  (wrist -> dactylion III)

Missing distal segments roll their mass up the anatomical chain (foot->shank->thigh,
hand->forearm->upper_arm) to the nearest observed proximal segment; an entirely
occluded chain lands on the trunk. Trunk and head mass is never redistributed. The
directly_observed_mass fraction and a CoMConfidence tier accompany every result.

Mass and com fractions live in anthropometric_parameters (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np

from skellyforge.kinematics.inertial.anthropometric_parameters import segment_inertial_parameters
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton

_SIDES: tuple[str, str] = (".L", ".R")

# de Leva segment -> standard-human SEGMENT (side-agnostic) whose own origin ->
# primary-axis-distal axis locates the de Leva segment CoM.
_DE_LEVA_1TO1: dict[str, str] = {
    "upper_arm": "upper_arm",
    "forearm": "lower_arm",
    "thigh": "upper_leg",
    "shank": "lower_leg",
    "foot": "foot",
}

# Composite de Leva segments -> (proximal_landmark, distal_landmark, sided).
# Landmarks are segment-layer landmarks (origins / distal landmarks), not keypoints.
_DE_LEVA_COMPOSITE: dict[str, tuple[str, str, bool]] = {
    "head": ("head_vertex", "neck_center", False),
    "trunk": ("neck_center", "hips_center", False),
    "hand": ("wrist", "hand_middle_finger_tip", True),
}

# Anatomical limb chains, distal -> proximal, as (de Leva name, segment key) tuples.
_SEGMENT_CHAINS: list[list[tuple[str, str]]] = [
    [("foot", f"{side}_foot"), ("shank", f"{side}_shank"), ("thigh", f"{side}_thigh")]
    for side in ("left", "right")
] + [
    [("hand", f"{side}_hand"), ("forearm", f"{side}_forearm"), ("upper_arm", f"{side}_upper_arm")]
    for side in ("left", "right")
]


class CoMConfidence(IntEnum):
    """Ordered confidence tier for a center-of-mass estimate (>= comparisons work)."""

    invalid = 0
    low = 1
    medium = 2
    high = 3


_CONFIDENCE_THRESHOLDS: list[tuple[float, CoMConfidence]] = [
    (0.90, CoMConfidence.high),
    (0.70, CoMConfidence.medium),
    (0.50, CoMConfidence.low),
]


def _confidence_from_mass(directly_observed: float) -> CoMConfidence:
    for threshold, tier in _CONFIDENCE_THRESHOLDS:
        if directly_observed >= threshold:
            return tier
    return CoMConfidence.invalid


@dataclass(frozen=True, slots=True)
class CenterOfMassResult:
    """Whole-body and per-segment center of mass."""

    total_body_com: np.ndarray
    segment_coms: dict[str, np.ndarray]
    segment_masses: dict[str, float]
    directly_observed_mass: float = 0.0
    confidence: CoMConfidence = CoMConfidence.invalid


def segment_center_of_mass(
    *, origin: np.ndarray, distal: np.ndarray, com_fraction: float
) -> np.ndarray:
    """A segment CoM at com_fraction from origin toward distal (de Leva)."""
    origin = np.asarray(origin, dtype=np.float64)
    distal = np.asarray(distal, dtype=np.float64)
    return origin + (distal - origin) * com_fraction


def calculate_center_of_mass(
    skeleton: HumanSkeleton,
    landmark_positions: dict[str, np.ndarray],
    *,
    sex: Literal["mean", "female", "male"] = "mean",
) -> CenterOfMassResult:
    """Segment-based whole-body center of mass with mass redistribution.

    Args:
        skeleton: the loaded standard-human skeleton. Each 1:1 de Leva segment is
            located by that segment own origin + primary-axis distal landmark.
        landmark_positions: solved/rigidified world positions keyed by standard-human
            LANDMARK name (the hydrated segment layer, not raw keypoints).
        sex: de Leva table selection (default: mean of female/male).

    Returns:
        CenterOfMassResult with whole-body CoM, per-segment CoM keyed by side-prefixed
        de Leva name, segment masses, directly-observed mass fraction, and confidence.
    """
    de_leva = segment_inertial_parameters(sex)

    segment_by_name: dict[str, list] = {}
    for segment in skeleton.segments:
        segment_by_name.setdefault(segment.name, []).append(segment)

    segment_coms: dict[str, np.ndarray] = {}
    segment_masses: dict[str, float] = {}

    def _add(de_leva_name: str, key: str, prox_name: str, dist_name: str) -> None:
        proximal = landmark_positions.get(prox_name)
        distal = landmark_positions.get(dist_name)
        if proximal is None or distal is None:
            return  # occlusion is data: this segment carries no CoM this frame
        info = de_leva[de_leva_name]
        segment_coms[key] = segment_center_of_mass(
            origin=proximal, distal=distal, com_fraction=info.com_fraction
        )
        segment_masses[key] = info.mass_fraction

    for de_leva_name, base_name in _DE_LEVA_1TO1.items():
        for suffix in _SIDES:
            matches = segment_by_name.get(base_name + suffix)
            if not matches:
                continue
            segment = matches[0]
            side = "left_" if suffix == ".L" else "right_"
            _add(
                de_leva_name,
                f"{side}{de_leva_name}",
                segment.origin_landmark.name,
                segment.primary_axis.target_landmark,
            )

    for de_leva_name, (prox, dist, sided) in _DE_LEVA_COMPOSITE.items():
        if sided:
            for prefix in ("left_", "right_"):
                _add(de_leva_name, f"{prefix}{de_leva_name}", prefix + prox, prefix + dist)
        else:
            _add(de_leva_name, de_leva_name, prox, dist)

    # --- mass-weighted whole-body CoM with redistribution along limb chains ---
    total = np.zeros(3, dtype=np.float64)
    directly_observed = 0.0
    orphan_mass = 0.0

    for chain in _SEGMENT_CHAINS:
        accumulated = 0.0
        for de_leva_name, seg_key in chain:
            seg_com = segment_coms.get(seg_key)
            seg_mass = de_leva[de_leva_name].mass_fraction
            if seg_com is not None:
                total += seg_com * (seg_mass + accumulated)
                directly_observed += seg_mass
                accumulated = 0.0
            else:
                accumulated += seg_mass
        if accumulated > 0.0:
            orphan_mass += accumulated

    trunk_com = segment_coms.get("trunk")
    if trunk_com is not None:
        trunk_mass = de_leva["trunk"].mass_fraction
        total += trunk_com * (trunk_mass + orphan_mass)
        directly_observed += trunk_mass

    head_com = segment_coms.get("head")
    if head_com is not None:
        head_mass = de_leva["head"].mass_fraction
        total += head_com * head_mass
        directly_observed += head_mass

    return CenterOfMassResult(
        total_body_com=total,
        segment_coms=segment_coms,
        segment_masses=segment_masses,
        directly_observed_mass=directly_observed,
        confidence=_confidence_from_mass(directly_observed),
    )
