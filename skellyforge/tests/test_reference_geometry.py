"""The T-pose reference geometry: right-handed everywhere, scales linearly."""

import numpy as np
import pytest

from skellyforge.skellymodels.standard_human.reference_geometry import (
    ReferenceGeometry,
)
from skellyforge.skellymodels.standard_human.segment_definition import AxisKind
from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)

_HEIGHT_MM = 1700.0


def _lengths(human) -> dict[str, float]:
    return {s.name: s.length_ratio * _HEIGHT_MM for s in human.segments}


def test_reference_basis_is_right_handed_for_every_segment_both_sides():
    human = compose_standard_human()
    geometry = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
    assert set(geometry.segments) == set(human.segment_names)
    for name, seg in geometry.segments.items():
        assert np.isclose(np.linalg.det(seg.basis), 1.0, atol=1e-9), name
        assert np.isclose(np.linalg.norm(seg.basis[0]), 1.0)
        assert np.isclose(np.linalg.norm(seg.basis[1]), 1.0)
        assert abs(np.dot(seg.basis[0], seg.basis[1])) < 1e-9, name
        assert np.isclose(seg.length, _lengths(human)[name])


def test_reference_geometry_scales_linearly_with_measured_lengths():
    human = compose_standard_human()
    single = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
    double = ReferenceGeometry.from_segments(
        list(human.segments), {n: 2 * l for n, l in _lengths(human).items()}
    )
    for name, seg in single.segments.items():
        assert np.allclose(seg.origin * 2.0, double.segments[name].origin)
        assert np.isclose(seg.length * 2.0, double.segments[name].length)


def test_no_segment_has_zero_length_in_the_reference_pose():
    human = compose_standard_human()
    geometry = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
    for name, seg in geometry.segments.items():
        assert seg.length > 0.0, name


def test_right_side_mirrors_positions_and_rebuilds_frames():
    human = compose_standard_human()
    geometry = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
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


def test_origin_landmark_positions_agree_with_segment_origins():
    # The consistency Fix-1 buys: a segment's rest origin and its origin
    # landmark's rest position are the same point. This is what makes the
    # solver's identity-at-T-pose contract hold.
    human = compose_standard_human()
    geometry = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
    for segment in human.segments:
        seg_geom = geometry.segments[segment.name]
        kp_pos = geometry.landmarks[segment.origin_landmark]
        assert np.allclose(seg_geom.origin, kp_pos, atol=1e-9), segment.name


def test_head_skull_landmarks_build_a_rest_map():
    # The head's 7-point skull set needs NO new rest-position logic: every name
    # except `nose` is another segment's (primary-axis/origin) landmark, so the
    # existing rest-map rules already place it. `nose` is the one off-chain
    # landmark — deliberately left out of the reference pose (the solver and
    # face bones supply it per frame).
    human = compose_standard_human()
    geometry = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
    for skull in ("head_center", "head_vertex", "left_eye", "right_eye", "left_ear", "right_ear"):
        assert skull in geometry.landmarks, skull
    assert "nose" not in geometry.landmarks  # off-chain, as before the reshape


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
    geometry = ReferenceGeometry.from_segments(list(human.segments), _lengths(human))
    head = geometry.segments["head"]
    # the head's exact axis is y (up); its approximate axis is z (anterior nose).
    # The rest frame is [x̂, ŷ, ẑ] = [+Y, +Z(up), +X(anterior)] — right-handed.
    assert np.allclose(head.basis[1], (0.0, 0.0, 1.0), atol=1e-9), head.basis[1]
    assert np.allclose(head.basis[2], (1.0, 0.0, 0.0), atol=1e-6), head.basis[2]


def _exact_axis(segment):
    return next(a for a in segment.axes if a.kind is AxisKind.EXACT)


def test_toward_child_rule_holds_for_every_body_segment():
    # Every body/hand segment's rest ŷ equals normalize(child_origin − origin) —
    # the VRM 1.0 humanoid rule (+Y toward the child bone). For leaves (no
    # child), the exact-axis direction IS the toward-child direction by
    # construction, so this reads the child chain where one exists.
    human = compose_standard_human()
    lengths = _lengths(human)
    geometry = ReferenceGeometry.from_segments(list(human.segments), lengths)
    by_name = {s.name: s for s in human.segments}
    for segment in human.segments:
        exact = _exact_axis(segment)
        if exact.axis != "y":
            continue  # face bones declare exact on z/x; not part of this rule
        child_origins = [
            geometry.segments[c.name].origin
            for c in human.get_children(segment.name)
        ]
        if not child_origins:
            # leaf — the exact-axis target is the toward-child point
            target = geometry.landmarks[exact.target_landmark]
            expected = target - geometry.segments[segment.name].origin
        else:
            # VRM: +Y toward the child bone — the FIRST child's origin
            expected = child_origins[0] - geometry.segments[segment.name].origin
        norm = np.linalg.norm(expected)
        if norm < 1e-9:
            continue
        expected = expected / norm
        assert np.allclose(
            geometry.segments[segment.name].basis[1], expected, atol=1e-6
        ), segment.name


def test_face_bones_rest_z_is_gaze():
    # The face bones (left_eye/right_eye/jaw) declare exact on z; their rest ẑ
    # is the gaze direction. The eyes' gaze is anterior (+X); the jaw's gaze
    # carries the authored downward chin offset (jaw→nose ≈ (0.217, 0, 0.976)).
    human = compose_standard_human()
    lengths = _lengths(human)
    geometry = ReferenceGeometry.from_segments(list(human.segments), lengths)
    by_name = {s.name: s for s in human.segments}
    for name in ("left_eye", "right_eye"):
        assert _exact_axis(by_name[name]).axis == "z", name
        assert np.allclose(geometry.segments[name].basis[2], (1.0, 0.0, 0.0), atol=1e-6), name
    # jaw: gaze points anterior-and-down, from the authored jaw offset
    assert _exact_axis(by_name["jaw"]).axis == "z"
    jaw_z = geometry.segments["jaw"].basis[2]
    assert np.isclose(jaw_z[1], 0.0, atol=1e-9)          # no lateral component
    assert jaw_z[0] > 0.0 and jaw_z[2] > 0.0              # anterior and up
    assert np.isclose(np.linalg.norm(jaw_z), 1.0)          # unit

