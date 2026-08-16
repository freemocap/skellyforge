"""The FreeMoCap face-part extension: nose, ears, mouth corners."""

import math

import numpy as np

from skellyforge.kinematics.coordinate_frame_ops import axis_index_and_sign
from skellyforge.skellymodels.standard_human.face_part import FACE_PART
from skellyforge.skellymodels.standard_human.reference_geometry import (
    _mirror,
    ReferenceGeometry,
)
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisKind,
    ParentAttachment,
)
from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)

_DETAIL_NAMES = {"nose", "left_ear", "right_ear", "left_mouth", "right_mouth"}

_HEIGHT_MM = 1700.0


def _by_name(name: str):
    return next(s for s in FACE_PART.segments if s.name == name)


def _exact_axis(seg):
    return next(a for a in seg.axes if a.kind is AxisKind.EXACT)


def _rest_dir_with_right_mirror(name: str) -> np.ndarray:
    """The segment's rest exact direction: the authored rest_direction, plus the
    right-side Y mirror."""
    seg = _by_name(name)
    exact = _exact_axis(seg)
    direction = np.asarray(exact.rest_direction, dtype=np.float64)
    if name.startswith("right_"):
        direction = _mirror(direction)
    return direction


def _reference_dirs() -> dict[str, np.ndarray]:
    """The rest exact direction for every composed segment, straight from the
    reference geometry (the ground truth the solver + stream schema use),
    read from the basis row NAMED by the segment's exact axis."""
    human = compose_standard_human()
    lengths = {s.name: s.length_ratio * _HEIGHT_MM for s in human.segments}
    ref = ReferenceGeometry.from_segments(list(human.segments), lengths)
    out = {}
    for name in human.segment_names:
        seg = next(s for s in human.segments if s.name == name)
        idx = axis_index_and_sign(_exact_axis(seg).axis)[0]
        out[name] = ref.segments[name].basis[idx]
    return out


def test_five_face_detail_segments_exist_with_authored_declaration():
    # All five are driven segments branching from the head ORIGIN, twistless,
    # with the nominal (direction-reference) ratio.
    for name in _DETAIL_NAMES:
        seg = _by_name(name)
        assert seg.parent == "head"
        assert seg.parent_attachment == ParentAttachment.ORIGIN
        assert seg.resolves_twist is False
        assert seg.rotation_limits is None

    assert _by_name("nose").origin_landmark == "head_center"
    assert _by_name("nose").axes[0].target_landmark == "nose"
    assert _by_name("left_ear").origin_landmark == "head_center"
    assert _by_name("left_ear").axes[0].target_landmark == "left_ear"
    assert _by_name("right_ear").origin_landmark == "head_center"
    assert _by_name("right_ear").axes[0].target_landmark == "right_ear"
    assert _by_name("left_mouth").origin_landmark == "left_mouth"
    assert _by_name("left_mouth").axes[0].target_landmark == "nose"
    assert _by_name("right_mouth").origin_landmark == "right_mouth"
    assert _by_name("right_mouth").axes[0].target_landmark == "nose"


def test_rest_directions_pin_the_head_axes():
    # The rest long axis (rest_direction, right side mirrored) yields the
    # authored head axes.
    assert np.allclose(_rest_dir_with_right_mirror("nose"), (1.0, 0.0, 0.0), atol=1e-9)
    assert np.allclose(_rest_dir_with_right_mirror("left_ear"), (0.0, 1.0, 0.0), atol=1e-9)
    assert np.allclose(_rest_dir_with_right_mirror("right_ear"), (0.0, -1.0, 0.0), atol=1e-9)


def test_mouth_corner_rest_directions_are_the_mapping_corner_offsets():
    # nose − left_mouth = (+0.2, −0.3, +0.35)·eye_width, normalized. The left
    # mouth corner sits at +Y so its vector has −Y; the right one mirrors it.
    norm = math.sqrt(0.2**2 + 0.3**2 + 0.35**2)
    expected_left = np.array([0.2, -0.3, 0.35]) / norm
    expected_right = np.array([0.2, 0.3, 0.35]) / norm

    assert np.allclose(_rest_dir_with_right_mirror("left_mouth"), expected_left, atol=1e-9)
    assert np.allclose(_rest_dir_with_right_mirror("right_mouth"), expected_right, atol=1e-9)


def test_reference_geometry_builds_the_correct_rest_directions():
    # The full reference geometry (not just rest_direction) must place the
    # ears at ±Y and the mouth corners at the mirrored corner→nose directions.
    dirs = _reference_dirs()
    assert np.allclose(dirs["nose"], (1.0, 0.0, 0.0), atol=1e-9)
    assert np.allclose(dirs["left_ear"], (0.0, 1.0, 0.0), atol=1e-9)
    assert np.allclose(dirs["right_ear"], (0.0, -1.0, 0.0), atol=1e-9)
    norm = math.sqrt(0.2**2 + 0.3**2 + 0.35**2)
    assert np.allclose(dirs["left_mouth"], np.array([0.2, -0.3, 0.35]) / norm, atol=1e-9)
    assert np.allclose(dirs["right_mouth"], np.array([0.2, 0.3, 0.35]) / norm, atol=1e-9)


def test_required_landmarks_includes_the_four_new_names_and_is_76():
    human = compose_standard_human()
    required = human.required_landmarks()
    assert {"left_ear", "right_ear", "left_mouth", "right_mouth"} <= required
    assert len(required) == 76
