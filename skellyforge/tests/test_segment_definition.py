import math
import numpy as np
import pytest
from skellyforge.kinematics.coordinate_frame_ops import build_segment_frame
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition, AxisKind, ParentAttachment, RotationLimits, SegmentDefinition,
)


def _foot_like(**overrides) -> SegmentDefinition:
    """A 3-point rigid body (ankle, foot_ball, heel): exact→foot_ball, approx→heel."""
    kwargs = dict(
        name="foot",
        parent="lower_leg",
        parent_attachment=ParentAttachment.DISTAL,
        rigid_points=("ankle", "foot_ball", "heel"),
        origin_keypoint="ankle",
        axes=(
            AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
            AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),
        ),
        rest_rotation=(0.0, 0.0, 0.0),
        rest_roll=0.0,
        length_ratio=0.026,
    )
    kwargs.update(overrides)
    return SegmentDefinition(**kwargs)


def test_required_keypoints_is_exactly_the_rigid_set():
    assert _foot_like().required_keypoints() == {"ankle", "foot_ball", "heel"}


def test_required_keypoints_omits_absent_twist_source():
    single = _foot_like(
        axes=(AxisDefinition("x", AxisKind.EXACT, "foot_ball"),)
    )
    assert single.required_keypoints() == {"ankle", "foot_ball", "heel"}


def test_resolves_twist_is_true_only_when_an_approximate_axis_is_declared():
    assert _foot_like().resolves_twist is True
    single = _foot_like(
        axes=(AxisDefinition("x", AxisKind.EXACT, "foot_ball"),)
    )
    assert single.resolves_twist is False


def test_axis_target_must_not_be_the_origin():
    with pytest.raises(ValueError, match="zero-length"):
        _foot_like(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "ankle"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),
            )
        )


def test_approximate_axis_target_must_not_be_the_origin():
    with pytest.raises(ValueError, match="zero-length"):
        _foot_like(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "ankle"),
            )
        )


def test_axis_target_must_be_a_rigid_point():
    # Every axis direction is origin → target; the target must be rigid on the
    # segment. This holds for BOTH kinds.
    with pytest.raises(ValueError, match="not in rigid_points"):
        _foot_like(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "wrist"),
            )
        )


def test_approximate_axis_target_must_be_a_rigid_point():
    # The core rule: a segment's frame is a function of its own points only —
    # the approximate target leaving the rigid set is a load-time error.
    with pytest.raises(ValueError, match="not in rigid_points"):
        _foot_like(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "big_toe"),
            )
        )


def test_must_declare_between_one_and_three_axes():
    with pytest.raises(ValueError, match="1–3 axes"):
        _foot_like(axes=())


def test_axis_names_must_be_distinct():
    with pytest.raises(ValueError, match="axis names must be distinct"):
        _foot_like(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
                AxisDefinition("x", AxisKind.APPROXIMATE, "heel"),
            )
        )


def test_axis_names_must_be_xyz():
    # the axis name is bounded by the ``Literal`` type, and the segment-level
    # validation additionally rejects any non-xyz name.
    with pytest.raises(ValueError, match="axis names must be in"):
        _foot_like(
            axes=(
                AxisDefinition("w", AxisKind.EXACT, "foot_ball"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),
            )
        )


def test_at_least_one_axis_must_be_exact():
    with pytest.raises(ValueError, match="at least one axis must be EXACT"):
        _foot_like(axes=(AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),))


def test_first_axis_must_be_exact():
    # The first declared axis is the segment's long axis (axes[0]); it must be
    # EXACT. An APPROXIMATE axis in the first slot reorders the long axis out
    # from under the solver, which reads axes[0] unconditionally.
    with pytest.raises(ValueError, match="first declared axis must be EXACT"):
        _foot_like(
            axes=(
                AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),
                AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
            )
        )


def test_rigid_points_must_have_at_least_two_keypoints():
    with pytest.raises(ValueError, match="at least 2"):
        _foot_like(rigid_points=("ankle",))


def test_rigid_points_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        _foot_like(rigid_points=("ankle", "ankle", "heel"))


def test_rigid_points_must_be_non_empty_strings():
    with pytest.raises(ValueError, match="non-empty"):
        _foot_like(rigid_points=("ankle", "", "heel"))


def test_origin_keypoint_must_be_in_the_rigid_set():
    with pytest.raises(ValueError, match="origin_keypoint"):
        _foot_like(origin_keypoint="wrist")


def test_name_must_be_snake_case():
    with pytest.raises(ValueError, match="snake_case"):
        _foot_like(name="upperArm")


def test_length_ratio_must_be_positive():
    with pytest.raises(ValueError, match="length_ratio"):
        _foot_like(length_ratio=0.0)


def test_length_ratio_must_be_finite():
    with pytest.raises(ValueError, match="length_ratio"):
        _foot_like(length_ratio=float("nan"))


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
        _foot_like(name="upper arm")


def test_parent_attachment_values_round_trip():
    assert ParentAttachment("origin") is ParentAttachment.ORIGIN
    assert ParentAttachment("distal") is ParentAttachment.DISTAL


# ── build_segment_frame dispatch ─────────────────────────────────────


def _two_axis_declaration(*, second_kind: AxisKind, second_dir: np.ndarray):
    return (
        AxisDefinition("x", AxisKind.EXACT, "distal"),
        AxisDefinition("y", second_kind, "twist"),
    ), {
        "origin": np.zeros(3, dtype=np.float64),
        "distal": np.array([1.0, 0.0, 0.0]),
        "twist": np.asarray(second_dir, dtype=np.float64),
    }


def test_single_exact_axis_returns_incomplete_frame():
    axes = (AxisDefinition("x", AxisKind.EXACT, "distal"),)
    positions = {
        "origin": np.zeros(3), "distal": np.array([1.0, 0.0, 0.0]),
    }
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is False
    assert basis is None


def test_two_exact_collinear_axes_raise():
    # the second EXACT direction is collinear with x̂ → a wrong declaration
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.EXACT, second_dir=np.array([2.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="collinear"):
        build_segment_frame(axes, positions, "origin")


def test_two_exact_non_collinear_axes_resolve():
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.EXACT, second_dir=np.array([0.0, 1.0, 0.0])
    )
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is True
    assert np.allclose(basis[0], [1.0, 0.0, 0.0])
    assert np.allclose(basis[2], [0.0, 0.0, 1.0])


def test_approximate_collinear_degrades_to_unresolved_not_raise():
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.APPROXIMATE, second_dir=np.array([2.0, 0.0, 0.0])
    )
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is False
    assert basis is None


def test_three_axes_use_third_only_for_z_sign():
    # third declaration's direction resolves ẑ's sign via dot-product.
    axes = (
        AxisDefinition("x", AxisKind.EXACT, "distal"),
        AxisDefinition("y", AxisKind.APPROXIMATE, "twist"),
        AxisDefinition("z", AxisKind.APPROXIMATE, "zref"),
    )
    positions = {
        "origin": np.zeros(3), "distal": np.array([1.0, 0.0, 0.0]),
        "twist": np.array([0.0, 1.0, 0.0]),
        "zref": np.array([0.0, 0.0, -1.0]),  # points against x̂×ŷ → flip ẑ
    }
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is True
    # x̂=(+X), ŷ=(+Y) → x̂×ŷ = +Z; the third direction (−Z) has negative dot →
    # ẑ is flipped to −Z to agree with the reference direction.
    assert np.allclose(basis[2], [0.0, 0.0, -1.0])


def test_builder_resolves_direction_from_the_origin_not_the_target():
    # The exact direction is origin → target; moving the origin (not just the
    # target) changes it. Left: target at [1,0,0] with origin at [0,0,0] →
    # +X. Right: same target but origin at [1,0,0] → the direction is now
    # zero-ish (degenerate) → unresolved.
    axes = (AxisDefinition("x", AxisKind.EXACT, "distal"),
            AxisDefinition("y", AxisKind.APPROXIMATE, "twist"))
    positions = {
        "origin": np.array([1.0, 0.0, 0.0]),
        "distal": np.array([1.0, 0.0, 0.0]),  # coincides with the origin
        "twist": np.array([0.0, 1.0, 0.0]),
    }
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is False
    assert basis is None


def test_builder_missing_origin_returns_unresolved():
    axes, positions = _two_axis_declaration(
        second_kind=AxisKind.APPROXIMATE, second_dir=np.array([0.0, 1.0, 0.0])
    )
    del positions["origin"]
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is False
    assert basis is None
