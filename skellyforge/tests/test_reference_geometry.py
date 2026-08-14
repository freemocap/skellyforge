"""The T-pose reference geometry: right-handed everywhere, scales linearly."""

import numpy as np
import pytest

from skellyforge.skellymodels.standard_human.reference_geometry import (
    build_reference_geometry,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)

_HEIGHT_MM = 1700.0


def _lengths(human) -> dict[str, float]:
    return {s.name: s.length_ratio * _HEIGHT_MM for s in human.segments}


def test_reference_basis_is_right_handed_for_every_segment_both_sides():
    human = compose_standard_human()
    geometry = build_reference_geometry(list(human.segments), _lengths(human))
    assert set(geometry.segments) == set(human.segment_names)
    for name, seg in geometry.segments.items():
        assert np.isclose(np.linalg.det(seg.basis), 1.0, atol=1e-9), name
        assert np.isclose(np.linalg.norm(seg.basis[0]), 1.0)
        assert np.isclose(np.linalg.norm(seg.basis[1]), 1.0)
        assert abs(np.dot(seg.basis[0], seg.basis[1])) < 1e-9, name
        assert np.isclose(seg.length, _lengths(human)[name])


def test_reference_geometry_scales_linearly_with_measured_lengths():
    human = compose_standard_human()
    single = build_reference_geometry(list(human.segments), _lengths(human))
    double = build_reference_geometry(
        list(human.segments), {n: 2 * l for n, l in _lengths(human).items()}
    )
    for name, seg in single.segments.items():
        assert np.allclose(seg.origin * 2.0, double.segments[name].origin)
        assert np.isclose(seg.length * 2.0, double.segments[name].length)


def test_no_segment_has_zero_length_in_the_reference_pose():
    human = compose_standard_human()
    geometry = build_reference_geometry(list(human.segments), _lengths(human))
    for name, seg in geometry.segments.items():
        assert seg.length > 0.0, name


def test_right_side_mirrors_positions_and_rebuilds_frames():
    human = compose_standard_human()
    geometry = build_reference_geometry(list(human.segments), _lengths(human))
    pairs = [
        (left_name, "right_" + left_name[len("left_"):])
        for left_name in human.segment_names
        if left_name.startswith("left_")
    ]
    assert pairs  # the body must actually have left/right pairs
    for left_name, right_name in pairs:
        left_geom = geometry.segments[left_name]
        right_geom = geometry.segments[right_name]
        # positions mirror: X and Z unchanged, Y negated
        assert np.isclose(left_geom.origin[0], right_geom.origin[0]), left_name
        assert np.isclose(left_geom.origin[1], -right_geom.origin[1]), left_name
        assert np.isclose(left_geom.origin[2], right_geom.origin[2]), left_name
        # both sides rebuilt right-handed — a mirrored basis would be det == -1
        assert np.isclose(np.linalg.det(left_geom.basis), 1.0), left_name
        assert np.isclose(np.linalg.det(right_geom.basis), 1.0), right_name


def test_origin_keypoint_positions_agree_with_segment_origins():
    # The consistency Fix-1 buys: a segment's rest origin and its origin
    # keypoint's rest position are the same point. This is what makes the
    # solver's identity-at-T-pose contract hold.
    human = compose_standard_human()
    geometry = build_reference_geometry(list(human.segments), _lengths(human))
    for segment in human.segments:
        seg_geom = geometry.segments[segment.name]
        kp_pos = geometry.keypoints[segment.origin_keypoint]
        assert np.allclose(seg_geom.origin, kp_pos, atol=1e-9), segment.name


def test_head_skull_rigid_points_build_a_rest_map():
    # The head's 7-point skull set needs NO new rest-position logic: every name
    # except `nose` is another segment's (long-axis/origin) keypoint, so the
    # existing rest-map rules already place it. `nose` is the one off-chain
    # keypoint — deliberately left out of the reference pose (the solver and
    # face bones supply it per frame).
    human = compose_standard_human()
    geometry = build_reference_geometry(list(human.segments), _lengths(human))
    for skull in ("head_center", "head_vertex", "left_eye", "right_eye", "left_ear", "right_ear"):
        assert skull in geometry.keypoints, skull
    assert "nose" not in geometry.keypoints  # off-chain, as before the reshape


def test_head_reference_forward_axis_is_anterior():
    # `nose` is the exact-axis target of SIX segments (the `nose` segment plus
    # both eyes, `jaw`, and both mouth corners). The schematic builder writes
    # the shared `nose` slot once per such segment, so it is last-writer-wins
    # (authoring order -> `right_mouth`), leaving `nose` in a NON-anterior
    # direction. The head's approximate axis must therefore resolve from its
    # authored anterior override, not that overwritten schematic position —
    # otherwise the head's reference frame is rolled off anatomical forward and
    # `identity == T-pose` breaks for the head at runtime (where a real
    # anterior nose is supplied). Long axis is +Z (up); forward is +X.
    human = compose_standard_human()
    geometry = build_reference_geometry(list(human.segments), _lengths(human))
    head = geometry.segments["head"]
    assert np.allclose(head.basis[0], (0.0, 0.0, 1.0), atol=1e-9), head.basis[0]
    assert np.allclose(head.basis[1], (1.0, 0.0, 0.0), atol=1e-6), head.basis[1]
