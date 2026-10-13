"""Regression tests for RTMPose marker aliases (freemocap/freemocap#886).

The RTMPose whole-body schema names the hand root ``hand_root`` (not
``wrist``) and the toes ``*_big_toe`` (not ``*_foot_index``). Before this
fix, ``Human.fix_hands_to_wrist`` and ``Human.put_skeleton_on_ground`` only
recognized the MediaPipe names, so both steps silently skipped on every
RTMPose recording (warning + ``return None``).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from skellyforge.skellymodels.managers.human import (
    HAND_WRIST_MARKERS,
    TOE_MARKERS_BY_SIDE,
    Human,
    _find_marker,
)
from skellyforge.skellymodels.models.tracking_model_info import RTMPoseModelInfo


def test_find_marker_prefers_first_candidate():
    assert _find_marker(["wrist", "hand_root"], HAND_WRIST_MARKERS) == "wrist"
    assert _find_marker(["hand_root"], HAND_WRIST_MARKERS) == "hand_root"
    assert _find_marker(["thumb1"], HAND_WRIST_MARKERS) is None
    assert (
        _find_marker(["left_big_toe"], TOE_MARKERS_BY_SIDE["left"]) == "left_big_toe"
    )
    assert (
        _find_marker(
            ["left_foot_index", "left_big_toe"], TOE_MARKERS_BY_SIDE["left"]
        )
        == "left_foot_index"
    )


def _make_rtmpose_human(n_frames: int = 10) -> Human:
    """Build a Human from synthetic RTMPose-ordered data (133 points)."""
    model_info = RTMPoseModelInfo()
    assert model_info.num_tracked_points == 133

    rng = np.random.default_rng(0)
    data = rng.random((n_frames, 133, 3)).astype(np.float64) * 1000.0

    # Per-frame drift so _find_still_frame sees varying velocity.
    data += np.arange(n_frames).reshape(-1, 1, 1) * 0.5

    human = Human.from_tracked_points_numpy_array(
        name="test",
        model_info=model_info,
        tracked_points_numpy_array=data,
    )
    return human


def test_fix_hands_to_wrist_accepts_hand_root(caplog):
    """RTMPose hands (``hand_root``) must be snapped, not skipped."""
    human = _make_rtmpose_human()

    # Place each hand root 20 mm from its body wrist so the snap is measurable.
    body_names = human.body.anatomical_structure.landmark_names
    left_wrist_idx = body_names.index("left_wrist")
    right_wrist_idx = body_names.index("right_wrist")
    body_xyz = human.body.xyz.as_array
    for side, wrist_idx in (("left", left_wrist_idx), ("right", right_wrist_idx)):
        hand = human.aspects[f"{side}_hand"]
        arr = hand.xyz.as_array.copy()
        arr[:, 0, :] = body_xyz[:, wrist_idx, :] + np.array([20.0, 0.0, 0.0])
        from skellyforge.skellymodels.models.trajectory import Trajectory
        from skellyforge.skellymodels.models.aspect import TrajectoryNames

        hand.add_trajectory(
            {
                TrajectoryNames.XYZ.value: Trajectory(
                    name=TrajectoryNames.XYZ.value,
                    array=arr,
                    landmark_names=hand.anatomical_structure.landmark_names,
                )
            }
        )

    with caplog.at_level(logging.WARNING):
        human.fix_hands_to_wrist()

    assert "Cannot fix hands to wrist" not in caplog.text

    # After the snap, each hand root coincides with its body wrist.
    body_xyz = (human.body.rigid_xyz or human.body.xyz).as_array
    for side, wrist_idx in (("left", left_wrist_idx), ("right", right_wrist_idx)):
        hand_root = human.aspects[f"{side}_hand"].xyz.as_array[:, 0, :]
        np.testing.assert_allclose(hand_root, body_xyz[:, wrist_idx, :], atol=1e-6)


def test_put_skeleton_on_ground_accepts_big_toe(caplog):
    """RTMPose feet (``*_big_toe``) must ground the skeleton, not skip."""
    human = _make_rtmpose_human()

    with caplog.at_level(logging.WARNING):
        human.put_skeleton_on_ground()

    assert "Missing foot markers" not in caplog.text
