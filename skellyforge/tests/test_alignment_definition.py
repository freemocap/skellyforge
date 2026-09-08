"""Alignment declarations resolve against the shipped skeleton, without name heuristics."""

from pathlib import Path

import pytest

from skellyforge.core.biomechanics.alignment_definition import AlignmentDefinition
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def test_human_alignment_resolves_model_objects() -> None:
    skeleton = SkeletonDefinition.from_default_yaml()
    definition = AlignmentDefinition.from_default_human(skeleton=skeleton)
    assert definition.body_regions[0] is skeleton.segments["skull"]
    assert len(definition.foot_contacts) == 6
    assert all(
        contact is skeleton.landmarks[contact.name]
        for contact in definition.foot_contacts
    )


def test_unknown_alignment_landmark_fails(tmp_path: Path) -> None:
    path = tmp_path / "alignment_definition.yaml"
    path.write_text(
        "body_regions: [skull]\nfoot_contacts: [unknown_landmark]\n", encoding="utf-8"
    )
    with pytest.raises(KeyError, match="unknown_landmark"):
        AlignmentDefinition.from_yaml(
            path=path, skeleton=SkeletonDefinition.from_default_yaml()
        )
