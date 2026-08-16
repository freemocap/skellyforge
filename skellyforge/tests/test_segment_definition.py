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
        landmarks=("ankle", "foot_ball", "heel"),
        origin_landmark="ankle",
        axes=(
            AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
            AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),
        ),

        length_ratio=0.026,
    )
    kwargs.update(overrides)
    return SegmentDefinition(**kwargs)


def test_rigid_with_parent_defaults_to_false():
    assert _foot_like().rigid_with_parent is False


def test_rigid_child_requires_a_parent():
    with pytest.raises(ValueError, match="rigid_with_parent requires a parent"):
        _foot_like(parent=None, rigid_with_parent=True)


def test_required_landmarks_is_exactly_the_rigid_set():
    assert _foot_like().required_landmarks() == {"ankle", "foot_ball", "heel"}


def test_required_landmarks_omits_absent_twist_source():
    single = _foot_like(
        axes=(AxisDefinition("x", AxisKind.EXACT, "foot_ball"),)
    )
    assert single.required_landmarks() == {"ankle", "foot_ball", "heel"}


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
    with pytest.raises(ValueError, match="not in landmarks"):
        _foot_like(
            axes=(
                AxisDefinition("x", AxisKind.EXACT, "foot_ball"),
                AxisDefinition("y", AxisKind.APPROXIMATE, "wrist"),
            )
        )


def test_approximate_axis_target_must_be_a_rigid_point():
    # The core rule: a segment's frame is a function of its own points only —
    # the approximate target leaving the rigid set is a load-time error.
    with pytest.raises(ValueError, match="not in landmarks"):
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


def test_at_least_one_axis_must_be_exact():
    with pytest.raises(ValueError, match="at least one axis must be EXACT"):
        _foot_like(axes=(AxisDefinition("y", AxisKind.APPROXIMATE, "heel"),))


def test_exact_axis_may_be_declared_on_any_name():
    # The exact axis may be declared on any of x/y/z; the tuple order is not the
    # construction order. These two constructions (exact on z, exact on y) are
    # both valid.
    z_exact = _foot_like(
        axes=(
            AxisDefinition("z", AxisKind.EXACT, "foot_ball"),
            AxisDefinition("x", AxisKind.APPROXIMATE, "heel"),
        )
    )
    y_exact = _foot_like(
        axes=(
            AxisDefinition("x", AxisKind.APPROXIMATE, "heel"),
            AxisDefinition("y", AxisKind.EXACT, "foot_ball"),
        )
    )
    assert z_exact.axes[0].axis == "z"
    assert y_exact.axes[0].axis == "x"  # approximate can precede exact in the tuple
    # both resolve the same exact target
    assert _exact_axis_name(z_exact) == "z"
    assert _exact_axis_name(y_exact) == "y"


def _exact_axis_name(seg) -> str:
    return next(a.axis for a in seg.axes if a.kind is AxisKind.EXACT)


def test_approximate_axis_may_precede_exact_in_the_tuple():
    # The construction order is the basis order (x, y, z), not the tuple order.
    # A y-exact + x-approximate segment authored with the approximate FIRST is
    # still valid (the exact axis is not required to fill any particular slot).
    seg = _foot_like(
        axes=(
            AxisDefinition("x", AxisKind.APPROXIMATE, "heel"),
            AxisDefinition("y", AxisKind.EXACT, "foot_ball"),
        )
    )
    assert _exact_axis_name(seg) == "y"
    assert seg.resolves_twist is True


def test_landmarks_must_have_at_least_two_landmarks():
    with pytest.raises(ValueError, match="at least 2"):
        _foot_like(landmarks=("ankle",))


def test_landmarks_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        _foot_like(landmarks=("ankle", "ankle", "heel"))


def test_landmarks_must_be_non_empty_strings():
    with pytest.raises(ValueError, match="non-empty"):
        _foot_like(landmarks=("ankle", "", "heel"))


def test_origin_landmark_must_be_in_the_rigid_set():
    with pytest.raises(ValueError, match="origin_landmark"):
        _foot_like(origin_landmark="wrist")


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


def test_approximate_before_exact_builds_projected_residual():
    # Hips/toes-style declaration: the APPROXIMATE axis (x) precedes the EXACT
    # axis (y) in basis order. The exact axis (+Z) is the defining direction and
    # must be the hard vector; the approximate (45° off-axis) must be
    # Gram-Schmidt-projected against it — NOT kept raw. This is the two-pass
    # construction: pass 1 builds exact-y hard, pass 2 projects approximate-x.
    axes = (
        AxisDefinition("x", AxisKind.APPROXIMATE, "twist"),
        AxisDefinition("y", AxisKind.EXACT, "distal"),
    )
    # exact-y target straight up (+Z); approximate-x target at 45° in the XZ
    # plane (off-axis, so its residual must be x̂ ≈ +X).
    off_axis = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    off_axis /= np.linalg.norm(off_axis)
    positions = {
        "origin": np.zeros(3, dtype=np.float64),
        "distal": np.array([0.0, 0.0, 1.0]),
        "twist": off_axis,
    }
    basis, resolved = build_segment_frame(axes, positions, "origin")
    assert resolved is True

    # The exact axis is hard +Z on its named row (y).
    assert np.allclose(basis[1], [0.0, 0.0, 1.0])
    # The approximate axis residual (x̂) is the projection of the 45° off-axis
    # target onto the plane orthogonal to ŷ=(+Z): +X (re-normalized), not the
    # raw 45° vector.
    assert np.allclose(basis[0], [1.0, 0.0, 0.0])
    # The third axis (ẑ) fills via the right-handed cross product x̂ × ŷ.
    # With x̂ = +X and ŷ = +Z (the exact axis maps the world +Z), that is
    # +X × +Z = −Y.
    assert np.allclose(basis[2], [0.0, -1.0, 0.0])

    # The reconstructed basis is orthonormal and right-handed.
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(3))
    assert np.isclose(np.linalg.det(basis), 1.0)
