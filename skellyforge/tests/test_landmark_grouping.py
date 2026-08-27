"""Landmark groups and connection groups: structure the model carries, so nobody parses names.

The point of these is negative as much as positive: with them, no consumer has to recover
"these four landmarks are a square" or "this point is part of the face" by splitting a
string. So the tests below check both that the structure survives the loader intact, and
that a grouping naming something that does not exist fails at load rather than at draw time.
"""

from __future__ import annotations

import pytest
import yaml

from skellyforge.core.skeleton.components.landmark_grouping import (
    LandmarkConnectionGroup,
    LandmarkGroup,
    build_landmark_connection_group,
    build_landmark_group,
)
from skellyforge.core.skeleton.loading import build_component
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def _component_with(*, groupings: str) -> dict:
    return yaml.safe_load(
        """
        segments:
          SPINE:
            reference_geometry:
              origin: BASE
              z_axis: {landmark: TOP, type: exact}
        landmarks:
          BASE: {definition: base, reference_frame: spine, local_position: [0, 0, 0]}
          TOP:  {definition: top,  reference_frame: spine, local_position: [0, 0, 1]}
        """
        + groupings
    )


# ── the objects ────────────────────────────────────────────────────────────


def test_a_group_that_names_nothing_is_refused() -> None:
    """An empty group is a leftover from an edit, not a grouping."""
    with pytest.raises(ValueError, match="is empty"):
        LandmarkGroup(name="face", landmark_names=())
    with pytest.raises(ValueError, match="no pairs"):
        LandmarkConnectionGroup(name="outline", pairs=())


def test_a_landmark_connected_to_itself_is_refused() -> None:
    with pytest.raises(ValueError, match="connected to"):
        LandmarkConnectionGroup(name="outline", pairs=(("nose", "nose"),))


def test_a_pair_that_is_not_a_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        LandmarkConnectionGroup(name="outline", pairs=(("nose", "chin", "ear"),))


def test_a_bare_string_tag_is_refused() -> None:
    """`tags: face` instead of `tags: [face]` is a YAML slip, so it is caught on that path.

    Tags are ordered — the palette resolves the first one it knows — so a single tag is
    still a list of one, and writing it as a scalar hides that ordering matters.
    """
    with pytest.raises(ValueError, match="must be a list of tag names"):
        build_landmark_group(
            name="face", entry={"landmark_names": ["nose"], "tags": "face"}
        )


def test_duplicate_tags_are_refused() -> None:
    with pytest.raises(ValueError, match="more than once"):
        LandmarkConnectionGroup(name="outline", pairs=(("a", "b"),), tags=("face", "face"))


def test_a_group_listing_a_landmark_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="more than once"):
        LandmarkGroup(name="face", landmark_names=("nose", "nose"))


def test_connection_groups_report_every_landmark_they_touch() -> None:
    group = LandmarkConnectionGroup(
        name="outline", pairs=(("a", "b"), ("b", "c"))
    )
    assert group.landmark_names == frozenset({"a", "b", "c"})


# ── the loader ─────────────────────────────────────────────────────────────


def test_groupings_load_with_names_and_tags_lowercased() -> None:
    """Tags are names too, so a palette lookup cannot miss on capitalization."""
    component = build_component(
        component=_component_with(
            groupings="""
        landmark_groups:
          ENDS:
            tags: [Ends, Spine]
            landmark_names: [BASE, TOP]
        landmark_connections:
          SPAN:
            pairs:
              - [BASE, TOP]
        """
        ),
        name="spine",
    )

    assert component.landmark_groups["ends"].landmark_names == ("base", "top")
    assert component.landmark_groups["ends"].tags == ("ends", "spine")
    assert component.landmark_connections["span"].pairs == (("base", "top"),)


def test_a_component_with_no_groupings_simply_has_none() -> None:
    """Groupings are optional — most components will never declare one."""
    component = build_component(component=_component_with(groupings=""), name="spine")
    assert component.landmark_groups == {}
    assert component.landmark_connections == {}


def test_an_unknown_top_level_section_is_refused() -> None:
    """A typo'd section name would otherwise be silently dropped."""
    with pytest.raises(ValueError, match="unexpected top-level sections"):
        build_component(
            component=_component_with(
                groupings="""
        landmark_grups:
          ENDS:
            landmark_names: [BASE]
        """
            ),
            name="spine",
        )


def test_a_grouping_naming_a_landmark_that_does_not_exist_fails_at_load() -> None:
    """Cross-component, so the skeleton is the first place that can catch it."""
    component = build_component(
        component=_component_with(
            groupings="""
        landmark_connections:
          SPAN:
            pairs:
              - [BASE, ELSEWHERE]
        """
        ),
        name="spine",
    )
    with pytest.raises(ValueError, match="name landmarks that do not exist"):
        SkeletonDefinition(
            name="t",
            landmarks=component.landmarks,
            segments=component.segments,
            landmark_connections=component.landmark_connections,
        )


def test_build_helpers_refuse_a_missing_member_list() -> None:
    with pytest.raises(ValueError, match="`landmark_names` must be a list"):
        build_landmark_group(name="face", entry={"tags": ["face"]})
    with pytest.raises(ValueError, match="`pairs` must be a list"):
        build_landmark_connection_group(name="outline", entry={"tags": ["face"]})


def test_build_helpers_refuse_unexpected_keys() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        build_landmark_group(
            name="face", entry={"landmark_names": ["nose"], "color": "#ffffff"}
        )


# ── the shipped human ──────────────────────────────────────────────────────


def test_the_standard_human_ships_skull_groupings() -> None:
    """The worked example: this is not a charuco feature.

    Asserted on the shipped model rather than a fixture, so deleting the authored groups
    fails here instead of quietly removing structure the frontend uses. The skull ships
    a GROUP and deliberately NO drawn connections: its sparse anatomical markers sit on
    a smooth surface, and edges between them read as a scribbled polygon over the face.
    """
    skeleton = SkeletonDefinition.from_default_yaml()

    assert "face_surface" in skeleton.landmark_groups
    assert skeleton.landmark_connections == {}

    # Every grouped name resolves to a real landmark, and the sided ones expanded.
    for group in skeleton.landmark_groups.values():
        for name in group.landmark_names:
            assert name in skeleton.landmarks, name
