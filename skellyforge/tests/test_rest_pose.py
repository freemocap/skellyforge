"""Tests for the rest pose (T-pose) loaded from YAML and resolved against a skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.type_overloads import FloatArray

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

EXPECTED_SEGMENT_COUNT: int = 61
EXPECTED_LANDMARK_COUNT: int = 124
GROUND_PLANE_TOLERANCE_MILLIMETRES: float = 0.5


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)


def _pose() -> RestPose:
    return RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=_skeleton())


def _distal_direction(*, pose: RestPose, name: str) -> FloatArray:
    """The world direction a segment's local +z (distal) points in the T-pose."""
    return pose.segment_orientations[name].rotate_vector(
        vector=np.array([0.0, 0.0, 1.0])
    )


def _written_variant(*, directory: Path, mutate) -> Path:
    """A copy of the shipped rest pose with one thing changed, written somewhere disposable."""
    document = yaml.safe_load(REST_POSE_YAML_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = directory / "variant_rest_pose.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# ── what the shipped rest pose is ─────────────────────────────────────


def test_the_rest_pose_resolves_every_segment_and_landmark() -> None:
    pose = _pose()
    skeleton = _skeleton()
    assert len(pose.segment_orientations) == len(skeleton.segments)
    assert set(pose.landmark_positions) == set(skeleton.landmarks)
    assert pose.root_segment_name == "pelvis"


def test_the_trunk_runs_straight_up() -> None:
    pose = _pose()
    for name in ("pelvis", "sacrolumbar", "thoracic", "cervical_spine", "skull"):
        np.testing.assert_allclose(
            _distal_direction(pose=pose, name=name), [0.0, 0.0, 1.0], atol=1e-6
        )


def test_the_arms_point_out_to_the_sides() -> None:
    pose = _pose()
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="left_upper_arm"), [-1.0, 0.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="right_upper_arm"), [1.0, 0.0, 0.0], atol=1e-6
    )
    # The lower arm and hand inherit the arm's orientation.
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="left_lower_arm"), [-1.0, 0.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="left_carpals"), [-1.0, 0.0, 0.0], atol=1e-6
    )


def test_the_legs_point_down_and_the_feet_forward() -> None:
    pose = _pose()
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="left_upper_leg"), [0.0, 0.0, -1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="left_lower_leg"), [0.0, 0.0, -1.0], atol=1e-6
    )
    # The foot slopes forward and down: the ankle sits above the ground, the ball reaches it.
    foot = _distal_direction(pose=pose, name="left_foot")
    assert foot[1] > 0.8 and foot[2] < -0.3
    np.testing.assert_allclose(
        _distal_direction(pose=pose, name="left_toes"), [0.0, 1.0, 0.0], atol=1e-3
    )


def test_the_heel_points_back_and_down() -> None:
    """The calcaneus is behind AND below the ankle - it is what the foot stands on.

    This asserts the sign that the authored quaternion used to get backwards, putting the
    heel bone above the ankle joint.
    """
    pose = _pose()
    for side in ("left", "right"):
        heel = _distal_direction(pose=pose, name=f"{side}_heel")
        assert heel[1] < -0.4, f"{side} heel must point backwards, got y={heel[1]:.3f}"
        assert heel[2] < -0.6, f"{side} heel must point downwards, got z={heel[2]:.3f}"


def test_both_feet_stand_on_one_flat_ground_plane() -> None:
    """Heel, ball and toe tip all reach the same height, on both sides.

    A standing foot rests on the calcaneus and the ball together. Checking it here is what
    ties the heel orientation, the heel length and the foot orientation into one claim -
    any of the three drifting alone breaks it.
    """
    positions = _pose().landmark_positions
    ground_landmarks = [
        f"{side}_{name}"
        for side in ("left", "right")
        for name in ("calcaneus", "ball", "toe_tip")
    ]
    heights = np.array([positions[name].array[2] for name in ground_landmarks])
    spread = float(heights.max() - heights.min())
    assert spread < GROUND_PLANE_TOLERANCE_MILLIMETRES, (
        f"the foot's ground contacts span {spread:.2f} mm: "
        f"{dict(zip(ground_landmarks, np.round(heights, 2)))}"
    )
    lowest_landmark_height = min(
        point.array[2] for point in positions.values()
    )
    assert heights.min() == pytest.approx(lowest_landmark_height, abs=1e-6), (
        "nothing should hang below the plane the feet stand on"
    )


def test_the_pelvis_sits_at_the_origin_and_the_lumbar_on_the_sacrum() -> None:
    pose = _pose()
    np.testing.assert_allclose(
        pose.segment_origins["pelvis"].array, [0.0, 0.0, 0.0], atol=1e-9
    )
    # sacrolumbar's origin IS the pelvis origin (hip center).
    np.testing.assert_allclose(
        pose.segment_origins["sacrolumbar"].array, [0.0, 0.0, 0.0], atol=1e-9
    )
    # upper_leg's origin is the left hemipelvis's hip socket.
    np.testing.assert_allclose(
        pose.segment_origins["left_upper_leg"].array, [-88.0, 0.0, 0.0], atol=1e-9
    )


def test_the_rest_pose_exposes_the_tree_it_was_built_from() -> None:
    """The parent tree, connect points and relative rotations are readable off the pose.

    Without these, every caller that needs the tree - the viewer, the round-trip test -
    has to parse `rest_pose.yaml` again, and a second parser is a second set of rules.
    """
    pose = _pose()
    assert pose.parents["pelvis"] is None
    assert pose.parents["left_lower_arm"] == "left_upper_arm"
    # An absent `connect_at` resolves to the segment's own origin landmark.
    assert pose.connect_ats["left_lower_arm"] == "left_elbow"
    assert pose.connect_ats["left_carpals"] == "left_wrist"
    assert set(pose.relative_orientations) == set(pose.segment_orientations)


# ── what the rest pose refuses ────────────────────────────────────────
#
# Topology refusals (parent / connect_at / root count) live in
# test_joint_definitions.py - the joints section owns those now. This file
# refuses only what a rest pose itself is responsible for.


def test_an_entry_naming_an_unknown_segment_is_rejected(tmp_path: Path) -> None:
    """A typo used to be silent, and silently moved that limb to the world origin."""

    def misspell_the_upper_arm(document: dict) -> None:
        document["segments"]["left_uppr_arm"] = document["segments"].pop("left_upper_arm")

    path = _written_variant(directory=tmp_path, mutate=misspell_the_upper_arm)
    with pytest.raises(ValueError, match="not in skeleton"):
        RestPose.from_yaml(path=path, skeleton=_skeleton())


def test_a_segment_with_no_entry_is_rejected(tmp_path: Path) -> None:
    def drop_the_skull(document: dict) -> None:
        document["segments"].pop("skull")

    path = _written_variant(directory=tmp_path, mutate=drop_the_skull)
    with pytest.raises(ValueError, match="needs a rest pose entry"):
        RestPose.from_yaml(path=path, skeleton=_skeleton())


def test_an_unknown_entry_key_is_rejected(tmp_path: Path) -> None:
    def add_a_typo_key(document: dict) -> None:
        document["segments"]["thoracic"]["orientaton"] = [1, 0, 0, 0]

    path = _written_variant(directory=tmp_path, mutate=add_a_typo_key)
    with pytest.raises(ValueError, match="unknown keys"):
        RestPose.from_yaml(path=path, skeleton=_skeleton())


def test_topology_keys_in_the_rest_pose_are_rejected(tmp_path: Path) -> None:
    """The joints section owns parent / connect_at now; a rest pose entry carrying
    them is stale authoring and must fail loudly rather than be silently obeyed."""

    def leave_stale_parent_fields(document: dict) -> None:
        document["segments"]["thoracic"] = {
            "parent": "sacrolumbar",
            "connect_at": "thoracolumbar_junction",
            "orientation": [1.0, 0.0, 0.0, 0.0],
        }

    path = _written_variant(directory=tmp_path, mutate=leave_stale_parent_fields)
    with pytest.raises(ValueError, match="joints:"):
        RestPose.from_yaml(path=path, skeleton=_skeleton())
