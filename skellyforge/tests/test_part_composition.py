import math
import pytest
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition, AxisKind, ParentAttachment, SegmentDefinition,
)
from skellyforge.skellymodels.standard_human.segment_parts import SegmentPart, compose_parts

def _hand_part() -> SegmentPart:
    return SegmentPart(
        name="hand",
        segments=(
            SegmentDefinition(
                name="hand", parent="lower_arm", parent_attachment=ParentAttachment.DISTAL,
                rigid_points=("wrist", "middle_finger_mcp"), origin_keypoint="wrist",
                axes=(
                    AxisDefinition("x", AxisKind.EXACT, "wrist", "middle_finger_mcp"),
                    AxisDefinition("y", AxisKind.APPROXIMATE, "wrist", "thumb_cmc"),
                ),
                rest_rotation=(0.0, math.radians(90.0), 0.0), rest_roll=math.radians(90.0),
                length_ratio=0.0505,
            ),
            SegmentDefinition(
                name="thumb_metacarpal", parent="hand", parent_attachment=ParentAttachment.ORIGIN,
                rigid_points=("thumb_cmc", "thumb_mcp"), origin_keypoint="thumb_cmc",
                axes=(AxisDefinition("x", AxisKind.EXACT, "thumb_cmc", "thumb_mcp"),),
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
    exact, approx = hand.axes
    assert (exact.from_keypoint, exact.to_keypoint) == ("left_wrist", "left_middle_finger_mcp")
    assert (approx.from_keypoint, approx.to_keypoint) == ("left_wrist", "left_thumb_cmc")

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
        rigid_points=("hips_center", "trunk_center"), origin_keypoint="hips_center",
        axes=(
            AxisDefinition("x", AxisKind.EXACT, "hips_center", "trunk_center"),
            AxisDefinition("y", AxisKind.APPROXIMATE, "hips_center", "right_hip"),
        ),
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


def test_midline_references_fall_back_to_unprefixed_names():
    # The limb part's shoulder attaches to the midline `upper_chest` and its twist
    # references the midline `neck_center`; the leg attaches to the midline `hips`.
    # Under prefix `left_` those become `left_upper_chest` / `left_neck_center` /
    # `left_hips`, none of which exist — name agreement resolves them back.
    midline = SegmentPart(name="midline", segments=(
        SegmentDefinition(
            name="hips", parent=None, parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=("hips_center", "trunk_center"), origin_keypoint="hips_center",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "hips_center", "trunk_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "hips_center", "right_hip"),
            ),
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.145,
        ),
        SegmentDefinition(
            name="upper_chest", parent="hips", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("mid_sternum", "neck_center"), origin_keypoint="mid_sternum",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "mid_sternum", "neck_center"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "mid_sternum", "right_shoulder"),
            ),
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.055,
        ),
    ))
    limb = SegmentPart(name="limb", segments=(
        SegmentDefinition(
            name="shoulder", parent="upper_chest", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("sternoclavicular", "shoulder"), origin_keypoint="sternoclavicular",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "sternoclavicular", "shoulder"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "sternoclavicular", "neck_center"),
            ),
            rest_rotation=(-math.pi / 2, 0.0, 0.0), rest_roll=0.0, length_ratio=0.103,
        ),
        SegmentDefinition(
            name="upper_leg", parent="hips", parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=("hip", "knee"), origin_keypoint="hip",
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "hip", "knee"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "hip", "ankle"),
            ),
            rest_rotation=(math.pi, 0.0, 0.0), rest_roll=0.0, length_ratio=0.245,
        ),
    ))
    composed = compose_parts([(midline, ""), (limb, "left_")])
    shoulder = next(s for s in composed if s.name == "left_shoulder")
    upper_leg = next(s for s in composed if s.name == "left_upper_leg")
    assert shoulder.parent == "upper_chest"
    _, approx = shoulder.axes
    assert (approx.from_keypoint, approx.to_keypoint) == ("left_sternoclavicular", "neck_center")
    assert upper_leg.parent == "hips"


from skellyforge.skellymodels.standard_human.body_part import compose_body_parts


# Hand-written deliberately: it is the independent authority the composed body
# is checked AGAINST (Task 6 keys the tracker-mapping completeness contract on
# this set). Deriving it from the body would make the test tautological — a
# typo in the authored data must fail it, not reproduce in the expected set.
BODY_KEYPOINT_SET = {
    # midline
    "hips_center", "trunk_center", "neck_center", "head_center", "nose",
    "mid_sternum", "head_vertex", "right_hip", "right_shoulder",
    # the head's 7-point skull rigid set (nose/head_vertex above) — eyes and
    # ears are rigid with the skull, hence body-required, but they are FACE
    # segments' authored long-axis/keypoint targets, named in the body only
    # because the head's rigid_points name them.
    "left_eye", "right_eye", "left_ear", "right_ear",
}
for _side in ("left_", "right_"):
    BODY_KEYPOINT_SET |= {
        f"{_side}sternoclavicular", f"{_side}shoulder", f"{_side}elbow",
        f"{_side}wrist", f"{_side}middle_finger_mcp", f"{_side}hip",
        f"{_side}knee", f"{_side}ankle", f"{_side}foot_ball", f"{_side}heel",
        f"{_side}big_toe", f"{_side}small_toe",
    }


BODY_PARENT_MAP = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "upper_chest": "chest",
    "neck": "upper_chest",
    "head": "neck",
    "left_shoulder": "upper_chest",
    "left_upper_arm": "left_shoulder",
    "left_lower_arm": "left_upper_arm",
    "left_upper_leg": "hips",
    "left_lower_leg": "left_upper_leg",
    "left_foot": "left_lower_leg",
    "left_toes": "left_foot",
    "right_shoulder": "upper_chest",
    "right_upper_arm": "right_shoulder",
    "right_lower_arm": "right_upper_arm",
    "right_upper_leg": "hips",
    "right_lower_leg": "right_upper_leg",
    "right_foot": "right_lower_leg",
    "right_toes": "right_foot",
}


def test_body_part_every_segment_reaches_the_root():
    body = compose_body_parts()
    assert len(body) == 20  # 6 midline + 7×2 limbs
    assert {s.name: s.parent for s in body} == BODY_PARENT_MAP
    for segment in body:
        seen = set()
        current = segment
        while current.parent is not None:
            assert current.name not in seen, f"cycle at {current.name}"
            seen.add(current.name)
            current = next(s for s in body if s.name == current.parent)
        assert current.name == "hips", f"{segment.name} terminates at {current.name}, not hips"


def test_body_part_declares_exactly_the_documented_keypoint_set():
    body = compose_body_parts()
    declared: set[str] = set()
    for segment in body:
        declared |= segment.required_keypoints()
    assert declared == BODY_KEYPOINT_SET


from skellyforge.skellymodels.standard_human.body_part import BODY_LIMB_PART, BODY_MIDLINE_PART
from skellyforge.skellymodels.standard_human.face_part import FACE_PART
from skellyforge.skellymodels.standard_human.hand_part import HAND_PART


def test_hand_part_has_sixteen_segments():
    assert len(HAND_PART.segments) == 16


def test_hand_composes_onto_the_body_by_name_agreement():
    composed = compose_parts([
        (BODY_MIDLINE_PART, ""),
        (BODY_LIMB_PART, "left_"),
        (BODY_LIMB_PART, "right_"),
        (HAND_PART, "left_"),
        (HAND_PART, "right_"),
    ])
    assert len(composed) == 52  # 20 body + 2×16 hand
    left_hand = next(s for s in composed if s.name == "left_hand")
    assert left_hand.parent == "left_lower_arm"
    assert any(s.name == "left_lower_arm" for s in composed)
    assert left_hand.origin_keypoint == "left_wrist"
    _, approx = left_hand.axes
    assert (approx.from_keypoint, approx.to_keypoint) == ("left_wrist", "left_thumb_cmc")


def test_face_part_declares_eight_face_segments():
    # The three VRM 1.0 face bones (eyes/jaw) plus the five FreeMoCap
    # face-detail segments (nose/ears/mouth corners) — all driven segments
    # branching from the head's ORIGIN, fully inside required_keypoints().
    assert {s.name for s in FACE_PART.segments} == {
        "left_eye", "right_eye", "jaw",
        "nose", "left_ear", "right_ear", "left_mouth", "right_mouth",
    }
    for segment in FACE_PART.segments:
        assert segment.parent == "head"
        assert segment.parent_attachment == ParentAttachment.ORIGIN
