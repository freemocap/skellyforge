import math
import numpy as np
import pytest
from skellyforge.kinematics.coordinate_frame_ops import build_segment_frame
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition, AxisKind, ParentAttachment, RotationLimits, SegmentDefinition,
)


def _upper_arm(**overrides) -> SegmentDefinition:
    kwargs = dict(
        name="upper_arm",
        parent="shoulder",
        parent_attachment=ParentAttachment.DISTAL,
        rigid_points=("shoulder", "elbow"),
        origin_keypoint="shoulder",
        axes=(
            AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),
            AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),
        ),
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
    single = _upper_arm(
        axes=(AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),)
    )
    assert single.required_keypoints() == {"shoulder", "elbow"}


def test_resolves_twist_is_true_only_when_an_approximate_axis_is_declared():
    assert _upper_arm().resolves_twist is True
    single = _upper_arm(
        axes=(AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),)
    )
    assert single.resolves_twist is False


def test_exact_axis_endpoints_must_differ():
    # This is the `neck`/`head` bug: two roles naming one point yields a zero-length
    # segment vector, and no orientation can be resolved from it.
    with pytest.raises(ValueError, match="zero-length"):
        _upper_arm(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "shoulder", "shoulder"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),
            )
        )


def test_approximate_axis_endpoints_must_differ():
    with pytest.raises(ValueError, match="zero-length"):
        _upper_arm(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "wrist", "wrist"),
            )
        )


def test_exact_axis_endpoints_must_be_rigid_points():
    # the exact axis is the segment's own geometry — its endpoints must be rigid
    # on the segment (not an external reference).
    with pytest.raises(ValueError, match="EXACT axis"):
        _upper_arm(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "shoulder", "wrist"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),
            )
        )


def test_approximate_axis_may_reference_a_keypoint_outside_the_rigid_set():
    # the upper arm's twist reference `wrist` is NOT rigid with the upper arm —
    # approximate-axis keypoints are a direction reference and may be external.
    seg = _upper_arm()
    assert seg.required_keypoints() == {"shoulder", "elbow", "wrist"}


def test_must_declare_between_one_and_three_axes():
    with pytest.raises(ValueError, match="1–3 axes"):
        _upper_arm(axes=())


def test_axis_names_must_be_distinct():
    with pytest.raises(ValueError, match="axis names must be distinct"):
        _upper_arm(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),
                AxisDefinition("x", AxisKind.APPROXIMATE, "shoulder", "wrist"),
            )
        )


def test_axis_names_must_be_xyz():
    # AxisDefinition self-validates its ``axis`` name now, so the invalid name
    # raises before the segment-level distinct-name/uniqueness check.
    with pytest.raises(ValueError, match="one of {'x','y','z'}"):
        _upper_arm(
            axes=(
                AxisDefinition("w", AxisKind.EXACT, "shoulder", "elbow"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),
            )
        )


def test_at_least_one_axis_must_be_exact():
    with pytest.raises(ValueError, match="at least one axis must be EXACT"):
        _upper_arm(
            axes=(AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),)
        )


def test_first_axis_must_be_exact():
    # The first declared axis is the segment's long axis (axes[0]); it must be
    # EXACT. An APPROXIMATE axis in the first slot reorders the long axis out
    # from under the solver, which reads axes[0] unconditionally.
    with pytest.raises(ValueError, match="first declared axis must be EXACT"):
        _upper_arm(
            axes=(
                AxisDefinition("y", AxisKind.APPROXIMATE, "shoulder", "wrist"),
                AxisDefinition("x", AxisKind.EXACT, "shoulder", "elbow"),
            )
        )


def test_rigid_points_must_have_at_least_two_keypoints():
    with pytest.raises(ValueError, match="at least 2"):
        _upper_arm(rigid_points=("shoulder",))


def test_rigid_points_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        _upper_arm(rigid_points=("shoulder", "shoulder"))


def test_rigid_points_must_be_non_empty_strings():
    with pytest.raises(ValueError, match="non-empty"):
        _upper_arm(rigid_points=("shoulder", ""))


def test_origin_keypoint_must_be_in_the_rigid_set():
    with pytest.raises(ValueError, match="origin_keypoint"):
        _upper_arm(origin_keypoint="wrist")


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


# ── build_segment_frame dispatch ─────────────────────────────────────


def _two_axis_declaration(*, second_kind: AxisKind, second_dir: np.ndarray):
    return (
        AxisDefinition("x", AxisKind.EXACT, "origin", "distal"),
        AxisDefinition("y", second_kind, "origin", "twist"),
    ), {
        "origin": np.zeros(3, dtype=np.float64),
        "distal": np.array([1.0, 0.0, 0.0]),
        "twist": np.asarray(second_dir, dtype=np.float64),
    }


def test_single_exact_axis_returns_incomplete_frame():
    axes = (AxisDefinition("x", AxisKind.EXACT, "origin", "distal"),)
    positions = {
        "origin": np.zeros(3), "distal": np.array([1.0, 0.0, 0.0]),
    }
    basis, resolved = build_segment_frame(axes, positions)
    assert resolved is False
    assert basis is None


def test_two_exact_collinear_axes_raise():
    # the second EXACT direction is collinear with x̂ → a wrong declaration
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.EXACT, second_dir=np.array([2.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="collinear"):
        build_segment_frame(axes, positions)


def test_two_exact_non_collinear_axes_resolve():
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.EXACT, second_dir=np.array([0.0, 1.0, 0.0])
    )
    basis, resolved = build_segment_frame(axes, positions)
    assert resolved is True
    assert np.allclose(basis[0], [1.0, 0.0, 0.0])
    assert np.allclose(basis[2], [0.0, 0.0, 1.0])


def test_approximate_collinear_degrades_to_unresolved_not_raise():
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.APPROXIMATE, second_dir=np.array([2.0, 0.0, 0.0])
    )
    basis, resolved = build_segment_frame(axes, positions)
    assert resolved is False
    assert basis is None


def test_three_axes_use_third_only_for_z_sign():
    # third declaration's direction resolves ẑ's sign via dot-product.
    axes = (
        AxisDefinition("x", AxisKind.EXACT, "origin", "distal"),
        AxisDefinition("y", AxisKind.APPROXIMATE, "origin", "twist"),
        AxisDefinition("z", AxisKind.APPROXIMATE, "origin", "zref"),
    )
    positions = {
        "origin": np.zeros(3), "distal": np.array([1.0, 0.0, 0.0]),
        "twist": np.array([0.0, 1.0, 0.0]),
        "zref": np.array([0.0, 0.0, -1.0]),  # points against x̂×ŷ → flip ẑ
    }
    basis, resolved = build_segment_frame(axes, positions)
    assert resolved is True
    # x̂=(+X), ŷ=(+Y) → x̂×ŷ = +Z; the third direction (−Z) has negative dot →
    # ẑ is flipped to −Z to agree with the reference direction.
    assert np.allclose(basis[2], [0.0, 0.0, -1.0])
