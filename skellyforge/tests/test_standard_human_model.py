"""The composed StandardHuman: 60 segments, validators, the driven contract."""

import pytest

from skellyforge.skellymodels.standard_human.human_bone_aliases import BONE_ALIASES
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    AxisKind,
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
            rigid_points=("a", "b"), origin_keypoint="a",
            axes=(AxisDefinition("x", AxisKind.EXACT, "b"),),
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
        SegmentDefinition(
            name="b", parent=None, parent_attachment=ParentAttachment.ORIGIN,
            rigid_points=("b", "a"), origin_keypoint="b",
            axes=(AxisDefinition("x", AxisKind.EXACT, "a"),),
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
    ))
    with pytest.raises(ValueError, match="exactly one root"):
        StandardHuman(name="x", parts=((two_roots, ""),))


def _root(name: str) -> SegmentDefinition:
    """A minimal valid root segment (parent is None)."""
    return SegmentDefinition(
        name=name, parent=None, parent_attachment=ParentAttachment.ORIGIN,
        rigid_points=(f"{name}_origin", f"{name}_distal"),
        origin_keypoint=f"{name}_origin",
        axes=(AxisDefinition("x", AxisKind.EXACT, f"{name}_distal"),),
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
            rigid_points=("a", "b"), origin_keypoint="a",
            axes=(AxisDefinition("x", AxisKind.EXACT, "b"),),
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
            rigid_points=("a", "c"), origin_keypoint="a",
            axes=(AxisDefinition("x", AxisKind.EXACT, "c"),),
            rest_rotation=(0.0, 0.0, 0.0), rest_roll=0.0, length_ratio=0.1,
        ),
        SegmentDefinition(
            name="b", parent="a", parent_attachment=ParentAttachment.DISTAL,
            rigid_points=("b", "c"), origin_keypoint="b",
            axes=(AxisDefinition("x", AxisKind.EXACT, "c"),),
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


def test_head_rigid_points_is_exactly_the_seven_name_skull_set():
    human = compose_standard_human()
    head = next(s for s in human.segments if s.name == "head")
    assert head.rigid_points == (
        "head_center", "head_vertex", "nose", "left_eye", "right_eye",
        "left_ear", "right_ear",
    )


def test_head_axes_exact_vertex_on_y_approximate_nose_on_z():
    human = compose_standard_human()
    head = next(s for s in human.segments if s.name == "head")
    exact = next(a for a in head.axes if a.kind is AxisKind.EXACT)
    approx = next(a for a in head.axes if a.kind is AxisKind.APPROXIMATE)
    # the head's exact axis is +Y toward its child-less apex (up); the
    # approximate axis is +Z toward the nose (the face direction).
    assert exact.axis == "y" and exact.target_keypoint == "head_vertex"
    assert approx.axis == "z" and approx.target_keypoint == "nose"
    assert head.resolves_twist is True


def test_foot_toes_hips_rigid_sets_are_the_full_bodies():
    human = compose_standard_human()
    by_name = {s.name: s for s in human.segments}
    assert by_name["hips"].rigid_points == (
        "hips_center", "trunk_center", "left_hip", "right_hip",
    )
    for side in ("left_", "right_"):
        foot = by_name[f"{side}foot"]
        toes = by_name[f"{side}toes"]
        assert foot.rigid_points == (f"{side}ankle", f"{side}foot_ball", f"{side}heel")
        assert toes.rigid_points == (f"{side}foot_ball", f"{side}big_toe", f"{side}small_toe")
        # the axis targets stay inside the segment's own rigid set
        for a in foot.axes:
            assert a.target_keypoint in foot.rigid_points
        for a in toes.axes:
            assert a.target_keypoint in toes.rigid_points


_DROP_LIST = (
    "upper_arm", "lower_arm", "shoulder", "neck", "upper_leg", "lower_leg",
)


def test_drop_list_segments_declare_exactly_one_exact_axis_and_no_twist():
    human = compose_standard_human()
    # the six 2-point limb/torso segments resolve roll through the damped
    # minimal tier: exactly one EXACT axis, no approximate, no twist source.
    for name in ("neck", "left_upper_arm", "right_upper_arm", "left_lower_arm",
                 "right_lower_arm", "left_shoulder", "right_shoulder",
                 "left_upper_leg", "right_upper_leg", "left_lower_leg",
                 "right_lower_leg"):
        seg = next(s for s in human.segments if s.name == name)
        assert len(seg.axes) == 1, name
        assert seg.axes[0].kind is AxisKind.EXACT, name
        assert seg.resolves_twist is False, name


def test_every_two_point_segment_exact_axis_target_is_its_distal_point():
    # for 2-point segments the exact axis target is the distal point: the
    # authored origin→long-axis direction. Spot-check a few across the body,
    # hand and face.
    human = compose_standard_human()
    expected = {
        "hips": ("trunk_center", "hips_center"),
        "left_upper_arm": ("left_elbow", "left_shoulder"),
        "left_foot": ("left_foot_ball", "left_ankle"),
        "left_middle_proximal": ("left_middle_finger_pip", "left_middle_finger_mcp"),
        "left_thumb_distal": ("left_thumb_tip", "left_thumb_ip"),
        "nose": ("nose", "head_center"),
        "left_mouth": ("nose", "left_mouth"),
    }
    for name, (target, origin) in expected.items():
        seg = next(s for s in human.segments if s.name == name)
        exact = next(a for a in seg.axes if a.kind is AxisKind.EXACT)
        assert exact.target_keypoint == target, name
        assert seg.origin_keypoint == origin, name
        assert exact.target_keypoint in seg.rigid_points, name


def test_hips_exact_is_trunk_center_approximate_is_right_hip():
    human = compose_standard_human()
    hips = next(s for s in human.segments if s.name == "hips")
    exact = next(a for a in hips.axes if a.kind is AxisKind.EXACT)
    approx = next(a for a in hips.axes if a.kind is AxisKind.APPROXIMATE)
    assert exact.target_keypoint == "trunk_center"
    assert approx.target_keypoint == "right_hip"
    assert hips.resolves_twist is True


def test_required_keypoints_is_unchanged_after_the_reshape():
    # The reshape renames fields and enriches the head's rigid set; the set of
    # keypoints the whole model requires a tracker to supply must be byte-for-
    # byte identical to the pre-reshape contract (76 names).
    human = compose_standard_human()
    assert sorted(human.required_keypoints()) == [
        "head_center", "head_vertex", "hips_center", "jaw", "left_ankle",
        "left_big_toe", "left_ear", "left_elbow", "left_eye", "left_foot_ball",
        "left_heel", "left_hip", "left_index_finger_dip", "left_index_finger_mcp",
        "left_index_finger_pip", "left_index_finger_tip", "left_knee",
        "left_middle_finger_dip", "left_middle_finger_mcp", "left_middle_finger_pip",
        "left_middle_finger_tip", "left_mouth", "left_pinky_dip", "left_pinky_mcp",
        "left_pinky_pip", "left_pinky_tip", "left_ring_finger_dip",
        "left_ring_finger_mcp", "left_ring_finger_pip", "left_ring_finger_tip",
        "left_shoulder", "left_small_toe", "left_sternoclavicular", "left_thumb_cmc",
        "left_thumb_ip", "left_thumb_mcp", "left_thumb_tip", "left_wrist",
        "mid_sternum", "neck_center", "nose", "right_ankle", "right_big_toe",
        "right_ear", "right_elbow", "right_eye", "right_foot_ball", "right_heel",
        "right_hip", "right_index_finger_dip", "right_index_finger_mcp",
        "right_index_finger_pip", "right_index_finger_tip", "right_knee",
        "right_middle_finger_dip", "right_middle_finger_mcp",
        "right_middle_finger_pip", "right_middle_finger_tip", "right_mouth",
        "right_pinky_dip", "right_pinky_mcp", "right_pinky_pip", "right_pinky_tip",
        "right_ring_finger_dip", "right_ring_finger_mcp", "right_ring_finger_pip",
        "right_ring_finger_tip", "right_shoulder", "right_small_toe",
        "right_sternoclavicular", "right_thumb_cmc", "right_thumb_ip",
        "right_thumb_mcp", "right_thumb_tip", "right_wrist", "trunk_center",
    ]
