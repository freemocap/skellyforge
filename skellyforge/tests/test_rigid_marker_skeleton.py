"""A rigid marked object is a skeleton, and every layer above it should already work.

These tests use a synthetic 3x2 grid of markers — the shape of a calibration board without
any charuco knowledge — to pin the claim the whole generic-skeleton effort rests on: a
one-segment skeleton needs no new machinery. It loads, it gets a rest pose without
authoring one, it hydrates to a pose AND a scale from the existing closed form, and its
scale means what its reference unit says it means.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.components.landmark_grouping import (
    LandmarkConnectionGroup,
    LandmarkGroup,
)
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.rigid_marker_skeleton import build_rigid_marker_skeleton
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution

# A 3x2 grid of markers one reference unit apart, on the z = 0 plane. `1.0` is the grid
# spacing, the way `1.0` is body height for the human — so a fitted scale of 54.0 means
# "the squares measure 54mm".
GRID_COLUMNS: int = 3
GRID_ROWS: int = 2


def _marker_positions() -> dict[str, np.ndarray]:
    return {
        f"marker_{row * GRID_COLUMNS + column}": np.array(
            [float(column), float(row), 0.0]
        )
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    }


def _grid_connection_group() -> LandmarkConnectionGroup:
    pairs: list[tuple[str, str]] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            index = row * GRID_COLUMNS + column
            if column < GRID_COLUMNS - 1:
                pairs.append((f"marker_{index}", f"marker_{index + 1}"))
            if row < GRID_ROWS - 1:
                pairs.append((f"marker_{index}", f"marker_{index + GRID_COLUMNS}"))
    return LandmarkConnectionGroup(name="grid", pairs=tuple(pairs), tags=("charuco_grid",))


def _board() -> SkeletonDefinition:
    return build_rigid_marker_skeleton(
        name="test_board",
        segment_name="board_plate",
        marker_positions=_marker_positions(),
        origin_marker_name="marker_0",
        primary_marker_name="marker_1",
        secondary_marker_name="marker_3",
        landmark_groups={
            "corners": LandmarkGroup(
                name="corners",
                landmark_names=tuple(sorted(_marker_positions())),
                tags=("charuco_corner",),
            )
        },
        landmark_connections={"grid": _grid_connection_group()},
    )


# ── it is a skeleton ───────────────────────────────────────────────────────


def test_a_rigid_marker_object_is_a_one_segment_skeleton() -> None:
    board = _board()

    assert len(board.segments) == 1
    assert len(board.landmarks) == GRID_COLUMNS * GRID_ROWS
    # No joints and no chains is a COMPLETE skeleton, not a deficient one.
    assert board.joints == {}
    assert board.chains == {}
    assert board.landmark_connections["grid"].tags == ("charuco_grid",)


def test_its_segment_rigid_fits_and_is_fully_specified() -> None:
    """Three non-collinear markers pin a full triad, so roll is measured, not conventional."""
    segment = _board().segments["board_plate"]
    assert segment.supports_rigid_fit
    assert segment.is_fully_specified


def test_the_origin_marker_sits_at_the_zero_of_the_segment_frame() -> None:
    """The invariant hydration relies on, whatever frame the caller's geometry was in."""
    board = build_rigid_marker_skeleton(
        name="offset_board",
        segment_name="board_plate",
        # Deliberately far from the origin: the builder recentres.
        marker_positions={
            name: position + np.array([10.0, -4.0, 7.0])
            for name, position in _marker_positions().items()
        },
        origin_marker_name="marker_0",
        primary_marker_name="marker_1",
        secondary_marker_name="marker_3",
    )
    np.testing.assert_allclose(
        board.landmarks["marker_0"].local_position.array, np.zeros(3), atol=1e-12
    )


# ── the defaults it relies on ──────────────────────────────────────────────


def test_it_gets_a_rest_pose_without_authoring_one() -> None:
    """A one-segment skeleton has a trivial tree, so `joints:` would be boilerplate."""
    board = _board()
    rest_pose = RestPose.default_for(skeleton=board)

    assert rest_pose.root_segment_name == "board_plate"
    assert rest_pose.parents == {"board_plate": None}
    np.testing.assert_allclose(
        rest_pose.segment_origins["board_plate"].array, np.zeros(3), atol=1e-12
    )
    # Identity rest orientation means the markers rest where they were authored.
    for marker_name, position in _marker_positions().items():
        np.testing.assert_allclose(
            rest_pose.landmark_positions[marker_name].array,
            position - _marker_positions()["marker_0"],
            atol=1e-12,
        )


def test_a_multi_segment_skeleton_with_no_joints_is_still_refused() -> None:
    """The default covers "no tree needed", not "tree forgotten".

    Two segments and no joints is two roots, which is not a tree — so there is no one right
    rest pose to default to, and it raises rather than silently rooting on whichever segment
    happened to be first.
    """
    first = _board()
    second = build_rigid_marker_skeleton(
        name="second_board",
        segment_name="second_plate",
        marker_positions={
            f"second_{name}": position for name, position in _marker_positions().items()
        },
        origin_marker_name="second_marker_0",
        primary_marker_name="second_marker_1",
        secondary_marker_name="second_marker_3",
    )
    two_segments = SkeletonDefinition(
        name="two",
        landmarks={**first.landmarks, **second.landmarks},
        segments={**first.segments, **second.segments},
    )
    with pytest.raises(ValueError, match="do not form one tree"):
        RestPose.default_for(skeleton=two_segments)


# ── it hydrates, with a scale ──────────────────────────────────────────────


def test_it_hydrates_to_a_pose_and_a_scale_from_the_existing_closed_form() -> None:
    """The whole claim: no new hydration code for a rigid object.

    The observed board is the authored one scaled by its square length and moved into the
    room, so the recovered scale must BE that square length — which is the number the user
    typed at calibration, and therefore a check on the reconstruction.
    """
    board = _board()
    square_length_mm = 54.0
    world_offset = np.array([120.0, -35.0, 900.0])
    observed = {
        name: Point.from_array(
            values=square_length_mm * board.landmarks[name].local_position.array
            + world_offset
        )
        for name in board.landmarks
    }

    pose = hydrate_skeleton(skeleton=board, observed=observed, require_all=True)
    segment_pose = pose.segment_poses["board_plate"]

    assert segment_pose.solved_by is PoseSolution.RIGID_FIT
    assert segment_pose.scale_estimate == pytest.approx(square_length_mm, rel=1e-9)
    np.testing.assert_allclose(segment_pose.origin.array, world_offset, atol=1e-8)


def test_a_partially_visible_object_still_hydrates() -> None:
    """Half the markers out of frame is ordinary, and three non-collinear ones suffice."""
    board = _board()
    square_length_mm = 54.0
    observed = {
        name: Point.from_array(
            values=square_length_mm * board.landmarks[name].local_position.array
        )
        for name in ("marker_0", "marker_1", "marker_3")
    }

    pose = hydrate_skeleton(skeleton=board, observed=observed, require_all=False)

    assert pose.segment_poses["board_plate"].scale_estimate == pytest.approx(
        square_length_mm, rel=1e-9
    )


# ── it refuses what it cannot answer ───────────────────────────────────────


def test_collinear_frame_markers_are_refused_when_the_object_is_built() -> None:
    """A definition mistake fails at build time, naming the object — not at first frame."""
    with pytest.raises(ValueError, match="collinear"):
        build_rigid_marker_skeleton(
            name="degenerate_board",
            segment_name="board_plate",
            marker_positions={
                "marker_0": np.array([0.0, 0.0, 0.0]),
                "marker_1": np.array([1.0, 0.0, 0.0]),
                "marker_2": np.array([2.0, 0.0, 0.0]),
            },
            origin_marker_name="marker_0",
            primary_marker_name="marker_1",
            secondary_marker_name="marker_2",
        )


def test_a_frame_marker_that_is_not_a_marker_is_refused() -> None:
    with pytest.raises(ValueError, match="not among its markers"):
        build_rigid_marker_skeleton(
            name="test_board",
            segment_name="board_plate",
            marker_positions=_marker_positions(),
            origin_marker_name="marker_0",
            primary_marker_name="marker_1",
            secondary_marker_name="nonexistent_marker",
        )


def test_too_few_markers_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        build_rigid_marker_skeleton(
            name="test_board",
            segment_name="board_plate",
            marker_positions={
                "marker_0": np.array([0.0, 0.0, 0.0]),
                "marker_1": np.array([1.0, 0.0, 0.0]),
            },
            origin_marker_name="marker_0",
            primary_marker_name="marker_1",
            secondary_marker_name="marker_1",
        )
