import math
import pytest
from skellyforge.skellymodels.standard_human.segment_definition import (
    ParentAttachment, SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import SegmentPart, compose_parts

def _hand_part() -> SegmentPart:
    return SegmentPart(
        name="hand",
        segments=(
            SegmentDefinition(
                name="hand", parent="lower_arm", parent_attachment=ParentAttachment.DISTAL,
                origin_keypoint="wrist", long_axis_keypoint="middle_finger_mcp",
                twist_keypoint="thumb_cmc",
                rest_rotation=(0.0, math.radians(90.0), 0.0), rest_roll=math.radians(90.0),
                length_ratio=0.0505,
            ),
            SegmentDefinition(
                name="thumb_metacarpal", parent="hand", parent_attachment=ParentAttachment.ORIGIN,
                origin_keypoint="thumb_cmc", long_axis_keypoint="thumb_mcp", twist_keypoint=None,
                rest_rotation=(0.0, math.radians(90.0), math.radians(-45.0)), rest_roll=0.0,
                length_ratio=0.0194,
            ),
        ),
    )

def test_instantiating_a_part_prefixes_segment_names():
    composed = compose_parts([(_hand_part(), "left_")])
    assert {s.name for s in composed} == {"left_hand", "left_thumb_metacarpal"}

def test_instantiating_a_part_prefixes_keypoint_names():
    composed = compose_parts([(_hand_part(), "left_")])
    hand = next(s for s in composed if s.name == "left_hand")
    assert hand.origin_keypoint == "left_wrist"
    assert hand.long_axis_keypoint == "left_middle_finger_mcp"
    assert hand.twist_keypoint == "left_thumb_cmc"

def test_instantiating_a_part_prefixes_parent_references_within_the_part():
    composed = compose_parts([(_hand_part(), "left_")])
    thumb = next(s for s in composed if s.name == "left_thumb_metacarpal")
    assert thumb.parent == "left_hand"

def test_a_parent_outside_the_part_is_still_prefixed_and_joins_by_name_agreement():
    # `lower_arm` lives in the body part; under prefix `left_` it becomes
    # `left_lower_arm`, which is exactly what the body part produces. Parts join
    # because their names coincide after prefixing — there is no attachment step.
    composed = compose_parts([(_hand_part(), "left_")])
    hand = next(s for s in composed if s.name == "left_hand")
    assert hand.parent == "left_lower_arm"

def test_the_same_part_instantiated_twice_yields_structurally_identical_sets():
    composed = compose_parts([(_hand_part(), "left_"), (_hand_part(), "right_")])
    left = sorted(s.name.removeprefix("left_") for s in composed if s.name.startswith("left_"))
    right = sorted(s.name.removeprefix("right_") for s in composed if s.name.startswith("right_"))
    assert left == right
    assert len(composed) == 4

def test_duplicate_segment_names_after_composition_raise():
    with pytest.raises(ValueError, match="duplicate segment name"):
        compose_parts([(_hand_part(), "left_"), (_hand_part(), "left_")])

def test_empty_prefix_leaves_midline_part_names_unchanged():
    # Task 3 authors the body's midline segments (hips, spine, chest, neck, head)
    # under an empty prefix — names and parents must come through untouched.
    composed = compose_parts([(_hand_part(), "")])
    assert {s.name for s in composed} == {"hand", "thumb_metacarpal"}
    hand = next(s for s in composed if s.name == "hand")
    assert hand.parent == "lower_arm"
    assert hand.origin_keypoint == "wrist"


def test_root_parent_none_survives_prefixing():
    root = SegmentDefinition(
        name="hips", parent=None, parent_attachment=ParentAttachment.ORIGIN,
        origin_keypoint="hips_center", long_axis_keypoint="trunk_center",
        twist_keypoint="right_hip",
        rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.145,
    )
    part = SegmentPart(name="midline", segments=(root,))
    composed = compose_parts([(part, "")])
    assert composed[0].parent is None
    # and a root authored inside a prefixed part would stay None too
    composed_prefixed = compose_parts([(part, "left_")])
    assert composed_prefixed[0].parent is None


def test_required_keypoints_are_fully_prefixed_after_composition():
    # The tracker-mapping boundary contract (Task 6) keys on these names —
    # a missed prefix here would silently break the model's required-keypoint set.
    composed = compose_parts([(_hand_part(), "left_")])
    hand = next(s for s in composed if s.name == "left_hand")
    thumb = next(s for s in composed if s.name == "left_thumb_metacarpal")
    assert hand.required_keypoints() == {"left_wrist", "left_middle_finger_mcp", "left_thumb_cmc"}
    assert thumb.required_keypoints() == {"left_thumb_cmc", "left_thumb_mcp"}
