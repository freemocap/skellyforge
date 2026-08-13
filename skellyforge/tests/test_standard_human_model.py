"""The composed StandardHuman: 60 segments, validators, the driven contract."""

import pytest

from skellyforge.skellymodels.standard_human.human_bone_aliases import BONE_ALIASES
from skellyforge.skellymodels.standard_human.segment_definition import (
    ParentAttachment,
    SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import SegmentPart
from skellyforge.skellymodels.standard_human.standard_human_model import (
    StandardHuman,
    compose_standard_human,
)


def test_composed_human_has_60_segments_matching_bone_aliases():
    human = compose_standard_human()
    assert len(human.segments) == 60
    assert set(human.segment_names) == set(BONE_ALIASES.keys())


def test_required_keypoints_include_the_face_bones():
    human = compose_standard_human()
    required = human.required_keypoints()
    assert {"left_eye", "right_eye", "jaw", "nose"} <= required
    assert {"left_ear", "right_ear", "left_mouth", "right_mouth"} <= required
    assert "left_wrist" in required  # driven: lower_arm/hand
    assert len(required) == 76


def test_two_roots_raise():
    two_roots = SegmentPart(name="two_roots", segments=(
        SegmentDefinition(
            name="a", parent=None, parent_attachment=ParentAttachment.ORIGIN,
            origin_keypoint="a", long_axis_keypoint="b", twist_keypoint=None,
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
        SegmentDefinition(
            name="b", parent=None, parent_attachment=ParentAttachment.ORIGIN,
            origin_keypoint="b", long_axis_keypoint="a", twist_keypoint=None,
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
    ))
    with pytest.raises(ValueError, match="exactly one root"):
        StandardHuman(name="x", parts=((two_roots, ""),))


def _root(name: str) -> SegmentDefinition:
    """A minimal valid root segment (parent is None)."""
    return SegmentDefinition(
        name=name, parent=None, parent_attachment=ParentAttachment.ORIGIN,
        origin_keypoint=f"{name}_origin", long_axis_keypoint=f"{name}_distal",
        twist_keypoint=None,
        rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
    )


def test_missing_parent_raises():
    # one root + an orphan whose parent (ghost) is not declared: the
    # missing-parent validator must fire (the root keeps the tree single-rooted
    # so the earlier root check passes).
    orphan = SegmentPart(name="orphan", segments=(
        _root("r"),
        SegmentDefinition(
            name="a", parent="ghost", parent_attachment=ParentAttachment.DISTAL,
            origin_keypoint="a", long_axis_keypoint="b", twist_keypoint=None,
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
    ))
    with pytest.raises(ValueError, match="not in the composed human"):
        StandardHuman(name="x", parts=((orphan, ""),))


def test_cycle_raises():
    # one root + a 2-cycle (a→b→a), all parents known: the root check passes and
    # the cycle detector must fire.
    cycle = SegmentPart(name="cycle", segments=(
        _root("r"),
        SegmentDefinition(
            name="a", parent="b", parent_attachment=ParentAttachment.DISTAL,
            origin_keypoint="a", long_axis_keypoint="c", twist_keypoint=None,
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
        SegmentDefinition(
            name="b", parent="a", parent_attachment=ParentAttachment.DISTAL,
            origin_keypoint="b", long_axis_keypoint="c", twist_keypoint=None,
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
    ))
    with pytest.raises(ValueError, match="cycle"):
        StandardHuman(name="x", parts=((cycle, ""),))


def test_hierarchy_accessors_agree():
    # doc 14 §7: segment_parents, children, and root-to-segment chains
    # describe the same tree.
    human = compose_standard_human()
    for segment in human.segments:
        chain = human.get_segment_chain(segment.name)
        assert chain[0].name == "hips"
        assert chain[-1].name == segment.name
        for parent, child in zip(chain, chain[1:]):
            assert child.parent == parent.name
            assert child in human.get_children(parent.name)
    # segment_parents agrees with the segments' own parent references
    assert human.segment_parents == {s.name: s.parent for s in human.segments}
