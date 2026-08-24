"""The mapping ↔ rest-pose round trip: every mapped landmark lands on its authored spot.

SkellyTracker's ``anatomical_offset`` ratios claim to reproduce SkellyForge's rest
positions ("identity == T-pose"). When one side moves without the other, mapped spine
and trunk landmarks drift centimetres off the geometry every other layer authors
against - and nothing downstream can tell. This test closes that loop: it drives each
body mapping's real ``TrackerMapping.apply`` with synthetic T-pose observations taken
from ``RestPose``, and asserts every produced landmark lands on its authored position.

Some landmarks are UNREACHABLE BY CONSTRUCTION from their entry's two-axis frame -
the authored point lies off the plane the frame spans - so they carry explicit
allowances here instead of silently failing or being rounded into green:
``_ALLOWED_ERROR_MM`` documents each one and why.

The module skips itself when skellytracker is not installed (dev dependency). The
regenerator that produces corrected ratios is
``skellyforge/scripts/generate_tracker_mapping_ratios.py``; keep its synthetic
observation table in sync with this file's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

pytest.importorskip("skellytracker", reason="skellytracker is a dev dependency")

from skellytracker.core.io.mapping_paths import (  # noqa: E402
    MEDIAPIPE_BODY_MAPPING,
    RTMPOSE_BODY_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping  # noqa: E402

_BODY_MAPPINGS = (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING)

_DEFAULT_TOLERANCE_MM = 2.0

# Landmarks whose authored rest positions cannot be expressed by their entry's
# frame (the point sits off the plane spanned by the declared axes + implicit
# third), documented instead of hidden. Values are measured residuals at T-pose.
_ALLOWED_ERROR_MM: dict[str, float] = {
    # The sternum points sit ~4.8mm superior to the shoulder-line plane the
    # trunk frame spans.
    "left_sternoclavicular": 5.0,
    "right_sternoclavicular": 5.0,
    "sternoclavicular_notch": 5.0,
    # The iliac crests sit ~6.4mm off that plane on the far side.
    "left_iliac_crest": 7.0,
    "right_iliac_crest": 7.0,
}

# Tracker keypoint name -> standard-human landmark whose REST position stands in
# for it at the synthetic T-pose. Keep in sync with
# scripts/generate_tracker_mapping_ratios.py.
_TRACKER_STANDINS: dict[str, str] = {
    "nose": "nose",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "left_eye_inner": "left_eye_inner",
    "right_eye_inner": "right_eye_inner",
    "left_eye_outer": "left_eye_outer",
    "right_eye_outer": "right_eye_outer",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "left_shoulder": "left_acromion",
    "right_shoulder": "right_acromion",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip_socket",
    "right_hip": "right_hip_socket",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_big_toe": "left_toe_tip",
    "right_big_toe": "right_toe_tip",
    "left_foot_index": "left_toe_tip",
    "right_foot_index": "right_toe_tip",
    "left_heel": "left_calcaneus",
    "right_heel": "right_calcaneus",
}


def _skeleton_and_rest_pose() -> tuple[SkeletonDefinition, RestPose]:
    skeleton = SkeletonDefinition.from_default_yaml()
    rest_pose = RestPose.from_default_yaml(skeleton=skeleton)
    return skeleton, rest_pose


def _synthetic_tpose_tracker_positions(
    rest_pose: RestPose,
) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for tracker_name, landmark_name in _TRACKER_STANDINS.items():
        try:
            positions[tracker_name] = rest_pose.landmark_positions[landmark_name].array
        except KeyError:
            missing.append(f"{tracker_name} -> {landmark_name}")
    if missing:
        raise ValueError(f"stand-in landmarks absent from the rest pose: {missing}")
    return positions


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS, ids=lambda p: p.name)
def test_every_mapped_landmark_lands_on_its_authored_rest_position(
    mapping_path: Path,
) -> None:
    skeleton, rest_pose = _skeleton_and_rest_pose()
    tracker_positions = _synthetic_tpose_tracker_positions(rest_pose)

    mapping = TrackerMapping.from_yaml(mapping_path)
    output = mapping.apply(tracker_positions=tracker_positions)

    offenders: list[str] = []
    for landmark_name, position in sorted(output.items()):
        authored_position = (
            rest_pose.landmark_positions[landmark_name].array
            if landmark_name in skeleton.landmarks
            else None
        )
        if authored_position is None:
            continue
        error_mm = float(np.linalg.norm(np.asarray(position) - authored_position))
        allowance = _ALLOWED_ERROR_MM.get(landmark_name, _DEFAULT_TOLERANCE_MM)
        if error_mm > allowance:
            offenders.append(
                f"{landmark_name}: {error_mm:.1f}mm (allowed {allowance:.1f}mm)"
            )

    assert not offenders, (
        f"{mapping_path.name}: these mapped landmarks miss their authored rest "
        f"positions - regenerate the ratios with "
        f"scripts/generate_tracker_mapping_ratios.py: " + ", ".join(offenders)
    )
