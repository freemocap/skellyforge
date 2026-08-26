"""Tests for the linkage layer's static face: joint definitions and topology.

The `joints:` section is the authoritative segment tree - these tests refuse the
malformed topologies it must never load, using disposable copies of the shipped
definition folder so an include-resolved variant is always loadable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

SHIPPED_DEFINITIONS_DIR: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
)

EXPECTED_JOINT_COUNT: int = 60  # segments minus the single root


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(
        path=SHIPPED_DEFINITIONS_DIR / "human_skeleton.yaml"
    )


def _variant_definitions_dir(directory: Path, mutate) -> Path:
    """A disposable copy of the shipped definition folder with one thing changed."""
    target = directory / "human_skeleton"
    shutil.copytree(SHIPPED_DEFINITIONS_DIR, target)
    skeleton_path = target / "human_skeleton.yaml"
    document = yaml.safe_load(skeleton_path.read_text(encoding="utf-8"))
    mutate(document)
    skeleton_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return target


def _load_variant(directory: Path, mutate) -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(
        path=_variant_definitions_dir(directory=directory, mutate=mutate)
        / "human_skeleton.yaml"
    )


# ── what the shipped joints are ───────────────────────────────────────


def test_the_shipped_topology_has_one_joint_per_non_root_segment() -> None:
    skeleton = _skeleton()
    assert len(skeleton.joints) == EXPECTED_JOINT_COUNT == len(skeleton.segments) - 1


def test_the_shipped_tree_roots_at_the_pelvis() -> None:
    skeleton = _skeleton()
    children = {joint.child.name for joint in skeleton.joints.values()}
    roots = sorted(set(skeleton.segments) - children)
    assert roots == ["pelvis"]


def test_every_joint_resolves_to_object_references() -> None:
    """After load, no joint still carries string names where objects belong."""
    for joint in _skeleton().joints.values():
        assert joint.parent.name in _skeleton().segments
        assert joint.child.name in _skeleton().segments
        assert joint.connect_at.name in joint.parent.landmarks


def test_default_convention_is_a_zyx_ball_with_axes_derived_names() -> None:
    skeleton = _skeleton()
    joint = skeleton.joints["left_lower_arm"]  # keyed by child segment for now
    assert joint.joint_type == "ball"
    assert joint.convention.sequence == "zyx"
    assert joint.convention.angle_names == ("z_angle", "y_angle", "x_angle")
    # Its connection point is the shared elbow landmark, owned by the parent.
    assert joint.connect_at.name == "left_elbow"
    assert joint.connect_at.name in joint.parent.landmarks


# ── what the joints section refuses ───────────────────────────────────


def test_a_connect_at_owned_by_no_parent_is_rejected(tmp_path: Path) -> None:
    def steal_a_landmark(document: dict) -> None:
        document["joints"]["sacrolumbar"]["connect_at"] = "head_vertex"

    with pytest.raises(ValueError, match="must be owned by the parent segment"):
        _load_variant(directory=tmp_path, mutate=steal_a_landmark)


def test_an_unknown_connect_at_landmark_is_rejected(tmp_path: Path) -> None:
    def misspell_the_landmark(document: dict) -> None:
        document["joints"]["sacrolumbar"]["connect_at"] = "sacrum_toppp"

    with pytest.raises(ValueError, match="is not a landmark"):
        _load_variant(directory=tmp_path, mutate=misspell_the_landmark)


def test_more_than_one_root_is_rejected(tmp_path: Path) -> None:
    def cut_the_thoracic_free(document: dict) -> None:
        document["joints"].pop("thoracic")

    with pytest.raises(ValueError, match="exactly one root segment"):
        _load_variant(directory=tmp_path, mutate=cut_the_thoracic_free)


def test_no_root_at_all_is_rejected(tmp_path: Path) -> None:
    def close_the_loop(document: dict) -> None:
        document["joints"]["pelvis_cycle"] = {
            "parent": "cervical_spine",
            "child": "pelvis",
            "connect_at": "craniocervical_junction",
        }

    with pytest.raises(ValueError, match="exactly one root segment"):
        _load_variant(directory=tmp_path, mutate=close_the_loop)


def test_a_segment_cannot_be_its_own_parent(tmp_path: Path) -> None:
    def self_joint(document: dict) -> None:
        document["joints"]["thoracic"]["parent"] = "thoracic"

    with pytest.raises(ValueError, match="joins segment .* to itself"):
        _load_variant(directory=tmp_path, mutate=self_joint)


def test_a_child_claimed_twice_is_rejected(tmp_path: Path) -> None:
    def second_elbow(document: dict) -> None:
        document["joints"]["left_forearm_duplicate"] = {
            "parent": "left_clavicle",
            "child": "left_lower_arm",
            "connect_at": "left_acromion",
        }

    with pytest.raises(ValueError, match="exactly one parent joint"):
        _load_variant(directory=tmp_path, mutate=second_elbow)


def test_an_unknown_parent_segment_is_rejected(tmp_path: Path) -> None:
    def phantom_parent(document: dict) -> None:
        document["joints"]["thoracic"]["parent"] = "left_uppr_arm"

    with pytest.raises(ValueError, match="does not have"):
        _load_variant(directory=tmp_path, mutate=phantom_parent)


def test_an_unknown_entry_key_is_rejected(tmp_path: Path) -> None:
    def typo_key(document: dict) -> None:
        document["joints"]["thoracic"]["conect_at"] = "thoracolumbar_junction"

    with pytest.raises(ValueError, match="unknown keys"):
        _load_variant(directory=tmp_path, mutate=typo_key)
