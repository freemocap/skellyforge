"""What an under-specified skeleton gets for free, and what it is still refused.

The sensible-defaults principle has a sharp edge, and these tests are where it lives: a
default resolves once at load because the model did not say and there is exactly one right
answer; anything a default cannot answer still raises, naming what is missing. Getting that
line wrong in either direction is how a format either grows boilerplate only one model can
fill, or starts quietly producing zeros.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.biomechanics.center_of_mass import (
    CenterOfMassDefinitions,
    compute_segment_coms,
    landmark_world_positions,
)
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.rigid_marker_skeleton import build_rigid_marker_skeleton
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

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


def _board(**overrides) -> SkeletonDefinition:
    return build_rigid_marker_skeleton(
        name="test_board",
        segment_name="board_plate",
        marker_positions=_marker_positions(),
        origin_marker_name="marker_0",
        primary_marker_name="marker_1",
        secondary_marker_name="marker_3",
        **overrides,
    )


# ── every skeleton has a centre of mass ────────────────────────────────────


def test_a_skeleton_that_declares_no_mass_model_still_has_a_centre_of_mass() -> None:
    """The default is the unweighted mean of each segment's landmarks."""
    board = _board()
    definitions = CenterOfMassDefinitions.default_for(skeleton=board)
    definitions.validate_against(skeleton=board)

    weights = definitions.get(name="board_plate").weights
    assert len(weights) == GRID_COLUMNS * GRID_ROWS
    assert {entry.weight for entry in weights} == {1.0 / (GRID_COLUMNS * GRID_ROWS)}


def test_the_default_centre_of_mass_lands_on_the_centroid_of_the_markers() -> None:
    """Unweighted means unweighted — checked against the geometric centre, in millimetres."""
    board = _board()
    square_length_mm = 54.0
    observed = {
        name: Point.from_array(
            values=square_length_mm * board.landmarks[name].local_position.array
        )
        for name in board.landmarks
    }
    pose = hydrate_skeleton(skeleton=board, observed=observed, require_all=True)
    world = landmark_world_positions(
        skeleton=board,
        pose=pose,
        segment_scales={"board_plate": pose.segment_poses["board_plate"].scale_estimate},
    )

    segment_coms = compute_segment_coms(
        definitions=CenterOfMassDefinitions.default_for(skeleton=board), world=world
    )

    expected = square_length_mm * np.mean(
        [
            position - _marker_positions()["marker_0"]
            for position in _marker_positions().values()
        ],
        axis=0,
    )
    np.testing.assert_allclose(segment_coms["board_plate"], expected, atol=1e-8)


def test_the_human_can_also_build_default_definitions() -> None:
    """The default is not a board special case — it is what "undeclared" means."""
    human = SkeletonDefinition.from_default_yaml()
    definitions = CenterOfMassDefinitions.default_for(skeleton=human)
    definitions.validate_against(skeleton=human)
    assert set(definitions.definitions) == set(human.segments)


# ── derived quantities are opt-in, and checked at load ─────────────────────


def test_a_skeleton_may_opt_into_nothing() -> None:
    assert _board().derived_quantities == frozenset()


def test_asking_for_an_unknown_derived_quantity_fails_at_load() -> None:
    with pytest.raises(ValueError, match="unknown derived quantities"):
        _board(derived_quantities=frozenset({"telekinesis"}))


def test_center_of_mass_is_refused_as_an_opt_in() -> None:
    """Listing it would make a universal property look optional."""
    with pytest.raises(ValueError, match="not an opt-in derived quantity"):
        _board(derived_quantities=frozenset({"center_of_mass"}))


def test_asking_for_inertia_without_a_mass_model_fails_at_load() -> None:
    """The sharp edge: a default cannot invent where a board's mass is, so it raises."""
    with pytest.raises(ValueError, match="declare no `anatomical_segment`"):
        _board(derived_quantities=frozenset({"inertia"}))

