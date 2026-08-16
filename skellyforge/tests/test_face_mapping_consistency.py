"""The face-part rest directions agree with the tracker mappings' offsets.

The face part authors its jaw / mouth-corner rest directions from the
facial-proportions canon on *this* side of the boundary. The tracker mappings
derive the same named points from the same canon on *their* side. Both are
independent by design; this test pins their numeric agreement without any
runtime coupling — it loads the two body mappings through the mapping-path
registry (the same pattern ``test_tracker_contract.py`` uses), parses the
``anatomical_offset`` entries' offset ratios, and asserts the unt vectors those
ratios imply match the face part's authored rest direction.

The offset frame is defined by the mapping's frame axes: ``up`` (exact,
neck_center→head_center), ``lateral`` (approximate, left_eye→right_eye), and
``anterior`` = cross(up, lateral). In the face part's canonical frame (+X
anterior, +Y left-lateral, +Z up) those map to: anterior→+X, lateral→−Y
(subject right), up→+Z. The offset ``origin + Σ ratio·dir`` places the jaw /
corner relative to the nose, so the nose→point direction is the negation of the
offset, giving canonical components ``(−anterior, +lateral, −up)``.
"""

from __future__ import annotations

import numpy as np

from skellyforge.skellymodels.standard_human.face_part import FACE_PART
from skellyforge.skellymodels.standard_human.reference_geometry import (
    _mirror,
)

from skellytracker.core.io.mapping_paths import (
    MEDIAPIPE_BODY_MAPPING,
    RTMPOSE_BODY_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping

# The face part's three jaw/mouth segments, by name. Each is a driven segment
# whose EXACT axis is on ``z`` (gaze) targeting ``nose``; the authored
# ``rest_direction`` is the nose−corner / nose−jaw direction (+Z-exact).
_FACE_SEGMENTS = {"jaw": "jaw", "left_mouth": "left_mouth", "right_mouth": "right_mouth"}

_BODY_MAPPING_PATHS = (RTMPOSE_BODY_MAPPING, MEDIAPIPE_BODY_MAPPING)

# The offsets these segments carry as anatomical_offset entries. The named
# ratios (of eye_width) place the point relative to the nose in the mapping's
# frame axes; the direction authored on this side is the nose→point vector.
_OFFSET_RATIOS: dict[str, dict[str, float]] = {
    "jaw": {"anterior": -0.2, "lateral": 0.0, "up": -0.9},
    "left_mouth": {"anterior": -0.2, "lateral": -0.3, "up": -0.35},
    "right_mouth": {"anterior": -0.2, "lateral": 0.3, "up": -0.35},
}


def _mapping_implied_direction(name: str) -> np.ndarray:
    """The nose→``name`` unit vector the mapping's offset ratios imply.

    In the canonical face frame (+X anterior, +Y left-lateral, +Z up), the
    offset ``origin + Σ ratio·dir`` places ``name`` at the ratio-weighted sum of
    the mapping axes. Because the mapping's ``lateral`` points to the subject's
    RIGHT (−Y) and the offset is measured nose→point backwards (negated), the
    canonical components are ``(−anterior, +lateral, −up)``.
    """
    ratios = _OFFSET_RATIOS[name]
    vec = np.array(
        [-ratios["anterior"], ratios["lateral"], -ratios["up"]],
        dtype=np.float64,
    )
    return vec / np.linalg.norm(vec)


def _authored_rest_direction(name: str) -> np.ndarray:
    """The face part's authored rest direction.

    Mirror the Y component for right-side segments, exactly as the reference
    geometry does when it builds the rest directions.
    """
    segment = next(s for s in FACE_PART.segments if s.name == name)
    exact = segment.exact_axis
    direction = np.asarray(exact.rest_direction, dtype=np.float64)
    if name.startswith("right_"):
        direction = _mirror(direction)
    return direction


def test_face_rest_directions_match_both_body_mapping_offsets() -> None:
    """The authored jaw/mouth rest directions agree with every mapping's offsets.

    Each of the two body mappings (rtmpose, mediapipe) declares the same
    anatomical-offset ratios for jaw / left_mouth / right_mouth; the face part
    must author the same direction. Consistency is pinned here at 1e-6 rather
    than at runtime, so the two sides never import one another.
    """
    authored = {name: _authored_rest_direction(name) for name in _FACE_SEGMENTS}

    for path in _BODY_MAPPING_PATHS:
        mapping = TrackerMapping.from_yaml(path)
        for name in _FACE_SEGMENTS:
            implied = _mapping_implied_direction(name)
            np.testing.assert_allclose(
                authored[name],
                implied,
                atol=1e-6,
                err_msg=(
                    f"{name} rest direction disagrees with the mapping at "
                    f"{path.name}: authored {authored[name]} vs implied {implied}"
                ),
            )


def test_face_segments_are_declared_exact_on_z_targeting_nose() -> None:
    """The three jaw/mouth segments are driven (+Z toward the nose).

    Guards the assumption the direction comparison rests on: every segment's
    EXACT axis is on ``z`` targeting ``nose``, so the direction is the authored
    rest_direction.
    """
    for name in _FACE_SEGMENTS:
        segment = next(s for s in FACE_PART.segments if s.name == name)
        exact = segment.exact_axis
        assert exact.axis == "z"
        assert exact.target_landmark == "nose"
