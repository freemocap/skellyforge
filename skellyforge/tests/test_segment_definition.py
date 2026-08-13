import math
import pytest
from skellyforge.skellymodels.standard_human.segment_definition import (
    ParentAttachment, RotationLimits, SegmentDefinition,
)


def _upper_arm(**overrides) -> SegmentDefinition:
    kwargs = dict(
        name="upper_arm",
        parent="shoulder",
        parent_attachment=ParentAttachment.DISTAL,
        origin_keypoint="shoulder",
        long_axis_keypoint="elbow",
        twist_keypoint="wrist",
        rest_rotation=(0.0, math.radians(90.0), 0.0),
        rest_roll=math.radians(90.0),
        length_ratio=0.186,
        rotation_limits=RotationLimits(x=(-135.0, 90.0), y=(-98.0, 180.0), z=(-97.0, 91.0)),
    )
    kwargs.update(overrides)
    return SegmentDefinition(**kwargs)


def test_required_keypoints_lists_all_three_roles():
    assert _upper_arm().required_keypoints() == {"shoulder", "elbow", "wrist"}


def test_required_keypoints_omits_absent_twist_source():
    assert _upper_arm(twist_keypoint=None).required_keypoints() == {"shoulder", "elbow"}


def test_resolves_twist_is_true_only_when_a_twist_keypoint_is_declared():
    assert _upper_arm().resolves_twist is True
    assert _upper_arm(twist_keypoint=None).resolves_twist is False


def test_origin_and_long_axis_keypoints_must_differ():
    # This is the `neck`/`head` bug: two roles naming one point yields a zero-length
    # segment vector, and no orientation can be resolved from it.
    with pytest.raises(ValueError, match="origin_keypoint and long_axis_keypoint"):
        _upper_arm(long_axis_keypoint="shoulder")


def test_twist_keypoint_must_differ_from_the_long_axis_keypoint():
    with pytest.raises(ValueError, match="twist_keypoint"):
        _upper_arm(twist_keypoint="elbow")


def test_name_must_be_snake_case():
    with pytest.raises(ValueError, match="snake_case"):
        _upper_arm(name="upperArm")


def test_length_ratio_must_be_positive():
    with pytest.raises(ValueError, match="length_ratio"):
        _upper_arm(length_ratio=0.0)


def test_length_ratio_must_be_finite():
    with pytest.raises(ValueError, match="length_ratio"):
        _upper_arm(length_ratio=float("nan"))


def test_rotation_limits_reject_inverted_bounds():
    with pytest.raises(ValueError, match="rotation_limits.x"):
        RotationLimits(x=(90.0, -90.0), y=(-98.0, 180.0), z=(-97.0, 91.0))


def test_rotation_limits_reject_nan_bounds():
    with pytest.raises(ValueError, match="rotation_limits.x"):
        RotationLimits(x=(float("nan"), 90.0), y=(-98.0, 180.0), z=(-97.0, 91.0))


def test_rotation_limits_accept_valid_bounds():
    limits = RotationLimits(x=(-135.0, 90.0), y=(-98.0, 180.0), z=(-97.0, 91.0))
    assert limits.x == (-135.0, 90.0)


def test_name_with_space_is_rejected():
    with pytest.raises(ValueError, match="snake_case"):
        _upper_arm(name="upper arm")


def test_parent_attachment_values_round_trip():
    assert ParentAttachment("origin") is ParentAttachment.ORIGIN
    assert ParentAttachment("distal") is ParentAttachment.DISTAL
