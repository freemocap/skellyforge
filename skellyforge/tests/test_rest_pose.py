"""Tests for the rest pose (T-pose) loaded from YAML and resolved against a skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.skeleton_parts.rest_pose import RestPose
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition

SKELETON_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)
REST_POSE_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "rest_pose.yaml"
)


def _pose() -> RestPose:
    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    return RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)


def _y_direction(pose: RestPose, name: str) -> np.ndarray:
    """The world direction a segment's local +y (distal) points in the T-pose."""
    return pose.segment_orientations[name].rotate_vector(np.array([0.0, 1.0, 0.0]))


def test_the_rest_pose_resolves_every_segment_and_landmark() -> None:
    pose = _pose()
    assert len(pose.segment_orientations) == 61
    assert len(pose.landmark_positions) == 169


def test_the_trunk_runs_straight_up() -> None:
    pose = _pose()
    for name in ("pelvis", "lumbar_spine", "chest", "cervical_spine", "skull"):
        np.testing.assert_allclose(_y_direction(pose, name), [0.0, 1.0, 0.0], atol=1e-6)


def test_the_arms_point_out_to_the_sides() -> None:
    pose = _pose()
    np.testing.assert_allclose(_y_direction(pose, "left_upper_arm"), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_y_direction(pose, "right_upper_arm"), [-1.0, 0.0, 0.0], atol=1e-6)
    # The lower arm and hand inherit the arm's orientation.
    np.testing.assert_allclose(_y_direction(pose, "left_lower_arm"), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_y_direction(pose, "left_carpals"), [1.0, 0.0, 0.0], atol=1e-6)


def test_the_legs_point_down_and_the_feet_forward() -> None:
    pose = _pose()
    np.testing.assert_allclose(_y_direction(pose, "left_upper_leg"), [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_y_direction(pose, "left_lower_leg"), [0.0, -1.0, 0.0], atol=1e-6)
    # The foot slopes forward and down: the ankle sits above the ground, the toes reach it.
    foot = _y_direction(pose, "left_foot")
    assert foot[2] > 0.8 and foot[1] < -0.3
    np.testing.assert_allclose(_y_direction(pose, "left_toes"), [0.0, 0.0, 1.0], atol=1e-3)


def test_the_heel_points_back() -> None:
    pose = _pose()
    heel = _y_direction(pose, "left_heel")
    assert heel[2] < -0.8 and heel[1] > 0.3


def test_the_pelvis_sits_at_the_origin_and_the_lumbar_on_the_sacrum() -> None:
    pose = _pose()
    np.testing.assert_allclose(pose.segment_origins["pelvis"].array, [0.0, 0.0, 0.0], atol=1e-9)
    # lumbar_spine's origin is the pelvis's sacrum_top local position.
    np.testing.assert_allclose(
        pose.segment_origins["lumbar_spine"].array, [0.0, 95.0, -35.0], atol=1e-9
    )
    # upper_leg's origin is the pelvis's left hip socket.
    np.testing.assert_allclose(
        pose.segment_origins["left_upper_leg"].array, [88.0, 0.0, 0.0], atol=1e-9
    )


def test_a_connect_at_must_belong_to_the_parent_segment() -> None:
    import yaml

    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    # A bogus connect_at (a skull landmark) for the lumbar spine must be rejected.
    document = yaml.safe_load(REST_POSE_YAML_PATH.read_text(encoding="utf-8"))
    document["segments"]["lumbar_spine"]["connect_at"] = "head_vertex"
    bad_path = REST_POSE_YAML_PATH.parent / "_bad_rest_pose.yaml"
    try:
        bad_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="must be owned by its parent"):
            RestPose.from_yaml(path=bad_path, skeleton=skeleton)
    finally:
        bad_path.unlink(missing_ok=True)
