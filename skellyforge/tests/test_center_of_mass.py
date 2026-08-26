"""Tests for center-of-mass definitions and their landmark-weighted computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.biomechanics.center_of_mass import (
    CenterOfMassDefinitions,
    compute_segment_coms,
    segment_com,
)
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

EXPECTED_SEGMENT_NAMES: int = 16


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)


def _definitions() -> CenterOfMassDefinitions:
    return CenterOfMassDefinitions.from_default_yaml()


def _world() -> dict[str, np.ndarray]:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    return {name: point.array for name, point in rest_pose.landmark_positions.items()}


def test_resolves_to_sixteen_sided_segment_names() -> None:
    assert len(_definitions().all_segment_names) == EXPECTED_SEGMENT_NAMES


def test_every_definition_has_weights_summing_to_one() -> None:
    for definition in _definitions().definitions.values():
        total = sum(entry.weight for entry in definition.weights)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_validates_against_the_shipped_skeleton() -> None:
    _definitions().validate_against(skeleton=_skeleton())


def test_abdomen_com_is_anterior_to_the_spine() -> None:
    world = _world()
    abdomen = segment_com(
        definition=_definitions().get(name="middle_trunk"), side=None, world=world
    )
    spine = world["chest_center"]
    assert abdomen[1] > spine[1]
    assert abdomen[1] > 0.0


def test_upper_arm_com_reproduces_the_de_leva_fraction() -> None:
    world = _world()
    shoulder = world["left_shoulder"]
    elbow = world["left_elbow"]
    com = segment_com(
        definition=_definitions().get(name="upper_arm"), side="left", world=world
    )
    expected = shoulder + 0.5772 * (elbow - shoulder)
    np.testing.assert_allclose(com, expected, atol=1e-9)


def test_com_landmarks_are_prefixed_by_side_for_bilateral_segments() -> None:
    world = _world()
    left = segment_com(
        definition=_definitions().get(name="thigh"), side="left", world=world
    )
    right = segment_com(
        definition=_definitions().get(name="thigh"), side="right", world=world
    )
    assert left[0] < 0.0 < right[0]


def test_rejects_weights_that_do_not_sum_to_one() -> None:
    document = {
        "segments": {
            "head_neck": {
                "proximal": "cervicothoracic_junction",
                "distal": "head_vertex",
                "landmarks": [{"landmark": "head_center", "weight": 0.9}],
            }
        }
    }
    with pytest.raises(ValueError):
        CenterOfMassDefinitions.from_document(document=document, source="test")


def test_rejects_mixing_bare_and_weighted_landmarks() -> None:
    document = {
        "segments": {
            "head_neck": {
                "proximal": "cervicothoracic_junction",
                "distal": "head_vertex",
                "landmarks": ["head_center", {"landmark": "chin", "weight": 0.5}],
            }
        }
    }
    with pytest.raises(ValueError):
        CenterOfMassDefinitions.from_document(document=document, source="test")
