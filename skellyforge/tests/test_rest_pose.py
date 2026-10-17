"""Tests for the rest-pose loader's resolution and validation behavior.

These exercise the code's own parsing and validation using the shipped rest pose as
input and mutated variants as failure cases. The specific geometry the rest pose authors
(segment names, orientations, coordinates) is data and is not asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

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


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)


def _pose() -> RestPose:
    return RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=_skeleton())


def _written_variant(*, directory: Path, mutate) -> Path:
    """A copy of the shipped rest pose with one thing changed, written somewhere disposable."""
    document = yaml.safe_load(REST_POSE_YAML_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = directory / "variant_rest_pose.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_the_rest_pose_resolves_every_segment_and_landmark() -> None:
    pose = _pose()
    skeleton = _skeleton()
    assert len(pose.segment_orientations) == len(skeleton.segments)
    assert set(pose.landmark_positions) == set(skeleton.landmarks)


def test_the_rest_pose_exposes_the_same_segment_set_it_resolves() -> None:
    pose = _pose()
    assert set(pose.relative_orientations) == set(pose.segment_orientations)


# ── what the rest pose refuses ────────────────────────────────────────


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
