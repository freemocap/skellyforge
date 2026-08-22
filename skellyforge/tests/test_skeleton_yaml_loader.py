"""Tests for loading skeleton component YAML into landmark and segment objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton_parts.skeleton_yaml_loader import (
    build_component,
    build_reference_frame_definition,
    expand_sided_entries,
    lowercase_names,
    resolve_includes,
)

PELVIS_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "components"
    / "pelvis.yaml"
)

HAND_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "components"
    / "hand.yml"
)

LEG_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "components"
    / "leg.yml"
)

FOOT_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "components"
    / "foot.yml"
)

SKELETON_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)


def _pelvis() -> SkeletonDefinition:
    return SkeletonDefinition.from_component_yaml(path=PELVIS_YAML_PATH, name="pelvis")


# ── $include ──────────────────────────────────────────────────────────


def test_include_is_equivalent_to_pasting_the_file_in(tmp_path: Path) -> None:
    # The whole contract: a document with an include means the same thing as the same
    # document with the included file's contents typed in its place.
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "arm.yaml").write_text("segments:\n  UPPER_ARM: {}\n")
    including = {"components": {"arm": {"$include": "parts/arm.yaml"}}}
    pasted = {"components": {"arm": {"segments": {"UPPER_ARM": {}}}}}
    assert (
        resolve_includes(node=including, base_directory=tmp_path, include_stack=())
        == pasted
    )


def test_include_paths_resolve_relative_to_the_including_file(tmp_path: Path) -> None:
    # `inner.yaml` includes `sibling.yaml` by bare name, so it must resolve next to
    # `inner.yaml` - not next to whoever included `inner.yaml`.
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "sibling.yaml").write_text("value: found_it\n")
    (nested / "inner.yaml").write_text("wrapped: {$include: sibling.yaml}\n")
    outer = {"$include": "nested/inner.yaml"}
    assert resolve_includes(node=outer, base_directory=tmp_path, include_stack=()) == {
        "wrapped": {"value": "found_it"}
    }


def test_include_mapping_may_carry_no_other_keys(tmp_path: Path) -> None:
    (tmp_path / "part.yaml").write_text("value: 1\n")
    with pytest.raises(ValueError, match="no other keys"):
        resolve_includes(
            node={"$include": "part.yaml", "extra": "ambiguous"},
            base_directory=tmp_path,
            include_stack=(),
        )


def test_include_cycles_are_caught(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("next: {$include: b.yaml}\n")
    (tmp_path / "b.yaml").write_text("back: {$include: a.yaml}\n")
    with pytest.raises(ValueError, match="cycle"):
        resolve_includes(
            node={"$include": "a.yaml"}, base_directory=tmp_path, include_stack=()
        )


def test_missing_include_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not name a file"):
        resolve_includes(
            node={"$include": "nope.yaml"}, base_directory=tmp_path, include_stack=()
        )


# ── lowercasing ───────────────────────────────────────────────────────


def test_names_aliases_and_references_are_lowercased_but_prose_is_not() -> None:
    component = yaml.safe_load(
        """
        segments:
          PELVIS:
            reference_geometry:
              origin: PELVIS_ORIGIN
              X_AXIS: {landmark: LEFT_HIP_SOCKET, type: exact}
        landmarks:
          PELVIS_ORIGIN:
            aliases: [HIPS_CENTER]
            definition: "Mean of the two hip sockets"
            reference_frame: PELVIS
            local_position: [0, 0, 0]
        """
    )
    lowercased = lowercase_names(node=component)
    assert sorted(lowercased["landmarks"]) == ["pelvis_origin"]
    assert lowercased["landmarks"]["pelvis_origin"]["aliases"] == ["hips_center"]
    assert lowercased["landmarks"]["pelvis_origin"]["reference_frame"] == "pelvis"
    geometry = lowercased["segments"]["pelvis"]["reference_geometry"]
    assert geometry["origin"] == "pelvis_origin"
    assert geometry["x_axis"]["landmark"] == "left_hip_socket"
    # Prose survives untouched.
    assert (
        lowercased["landmarks"]["pelvis_origin"]["definition"]
        == "Mean of the two hip sockets"
    )


# ── sidedness ─────────────────────────────────────────────────────────


def _sided_component() -> dict[str, object]:
    return lowercase_names(
        node=yaml.safe_load(
            """
            segments:
              UPPER_ARM:
                sided: true
                reference_geometry:
                  origin: SHOULDER
                  y_axis: {landmark: ELBOW, type: exact}
            landmarks:
              SHOULDER:
                sided: true
                aliases: [SHO]
                definition: "Glenohumeral joint center"
                reference_frame: UPPER_ARM
                local_position: [180, 1400, 0]
              ELBOW:
                sided: true
                definition: "Elbow joint center"
                reference_frame: UPPER_ARM
                local_position: [180, 1120, 0]
            """
        )
    )


def test_sided_entries_become_a_left_and_a_right() -> None:
    expanded = expand_sided_entries(component=_sided_component())
    assert sorted(expanded["segments"]) == ["left_upper_arm", "right_upper_arm"]
    assert sorted(expanded["landmarks"]) == [
        "left_elbow",
        "left_shoulder",
        "right_elbow",
        "right_shoulder",
    ]


def test_the_right_side_is_mirrored_across_the_sagittal_plane() -> None:
    expanded = expand_sided_entries(component=_sided_component())
    assert expanded["landmarks"]["left_shoulder"]["local_position"] == [180.0, 1400.0, 0.0]
    assert expanded["landmarks"]["right_shoulder"]["local_position"] == [-180.0, 1400.0, 0.0]


def test_references_inside_a_sided_entry_resolve_to_the_same_side() -> None:
    expanded = expand_sided_entries(component=_sided_component())
    left = expanded["segments"]["left_upper_arm"]["reference_geometry"]
    right = expanded["segments"]["right_upper_arm"]["reference_geometry"]
    assert left["origin"] == "left_shoulder"
    assert left["y_axis"]["landmark"] == "left_elbow"
    assert right["origin"] == "right_shoulder"
    assert right["y_axis"]["landmark"] == "right_elbow"
    assert expanded["landmarks"]["right_shoulder"]["reference_frame"] == "right_upper_arm"
    assert expanded["landmarks"]["right_shoulder"]["aliases"] == ["right_sho"]


def test_a_file_level_sided_flag_sides_every_entry() -> None:
    component = lowercase_names(
        node=yaml.safe_load(
            """
            sided: true
            segments:
              HEEL:
                reference_geometry:
                  origin: ANKLE
                  y_axis: {landmark: CALCANEUS, type: exact}
            landmarks:
              ANKLE:
                definition: "Ankle joint center"
                reference_frame: HEEL
                local_position: [90, 80, 0]
              CALCANEUS:
                definition: "Calcaneal tuberosity"
                reference_frame: HEEL
                local_position: [90, 30, -50]
            """
        )
    )
    expanded = expand_sided_entries(component=component)
    assert sorted(expanded["segments"]) == ["left_heel", "right_heel"]
    assert sorted(expanded["landmarks"]) == [
        "left_ankle",
        "left_calcaneus",
        "right_ankle",
        "right_calcaneus",
    ]


def test_a_sided_landmark_may_span_negative_x() -> None:
    # Fan-shaped structures (the hand) put the thumb on +x and the pinky on -x within one
    # side. Negative x is legitimate; the right side is still the mirror of the left.
    component = _sided_component()
    component["landmarks"]["shoulder"]["local_position"] = [-180, 1400, 0]
    expanded = expand_sided_entries(component=component)
    assert expanded["landmarks"]["left_shoulder"]["local_position"] == [-180.0, 1400.0, 0.0]
    assert expanded["landmarks"]["right_shoulder"]["local_position"] == [180.0, 1400.0, 0.0]


def test_an_unsided_entry_keeps_its_explicit_side_references() -> None:
    # The pelvis is one body but its x axis points at the LEFT hip, so it names that
    # side outright rather than inheriting one.
    pelvis = _pelvis().segments["pelvis"]
    assert pelvis.frame_definition.primary_point_name == "left_hip_socket"


# ── reference_geometry -> ReferenceFrameDefinition ────────────────────


def test_exact_becomes_primary_and_approximate_becomes_secondary() -> None:
    definition = build_reference_frame_definition(
        segment_name="pelvis",
        reference_geometry={
            "origin": "pelvis_origin",
            "x_axis": {"landmark": "left_hip_socket", "type": "exact"},
            "y_axis": {"landmark": "sacrum_top", "type": "approximate"},
        },
    )
    assert definition.primary_axis is SpatialAxis.X
    assert definition.primary_point_name == "left_hip_socket"
    assert definition.secondary_axis is SpatialAxis.Y
    assert definition.secondary_point_name == "sacrum_top"
    assert definition.is_fully_specified


def test_negate_selects_the_negative_half_axis() -> None:
    definition = build_reference_frame_definition(
        segment_name="lumbar_spine",
        reference_geometry={
            "origin": "sacrum_top",
            "y_axis": {"landmark": "spine_t12", "type": "exact"},
            "z_axis": {"landmark": "xiphoid_process", "type": "approximate", "negate": True},
        },
    )
    assert definition.secondary_axis is SpatialAxis.NEGATIVE_Z


def test_geometry_with_no_approximate_axis_is_underspecified() -> None:
    definition = build_reference_frame_definition(
        segment_name="upper_arm",
        reference_geometry={
            "origin": "shoulder",
            "y_axis": {"landmark": "elbow", "type": "exact"},
        },
    )
    assert not definition.is_fully_specified
    assert definition.point_names == ("shoulder", "elbow")


@pytest.mark.parametrize(
    "reference_geometry,message",
    [
        ({"y_axis": {"landmark": "elbow", "type": "exact"}}, "needs an `origin`"),
        ({"origin": "a", "w_axis": {"landmark": "b", "type": "exact"}}, "unknown"),
        ({"origin": "a", "y_axis": {"landmark": "b", "type": "roughly"}}, "must be"),
        ({"origin": "a", "y_axis": {"landmark": "b", "type": "approximate"}}, "type: exact"),
        (
            {
                "origin": "a",
                "y_axis": {"landmark": "b", "type": "exact"},
                "z_axis": {"landmark": "c", "type": "exact"},
            },
            "exactly one axis",
        ),
    ],
    ids=["no-origin", "unknown-axis", "unknown-type", "no-exact", "two-exact"],
)
def test_malformed_reference_geometry_raises(
    reference_geometry: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_reference_frame_definition(
            segment_name="segment", reference_geometry=reference_geometry
        )


# ── the real pelvis file ──────────────────────────────────────────────


def test_the_shipped_pelvis_yaml_loads() -> None:
    pelvis = _pelvis()
    assert sorted(pelvis.segments) == ["pelvis"]
    # Five midline landmarks, plus five sided ones expanded into a left and a right.
    assert len(pelvis.landmarks) == 15
    assert pelvis.underspecified_segment_names == ()


def test_the_pelvis_segment_is_fully_specified() -> None:
    pelvis = _pelvis().segments["pelvis"]
    assert pelvis.is_fully_specified
    assert pelvis.frame_definition.origin_point_name == "pelvis_origin"
    assert pelvis.frame_definition.primary_axis is SpatialAxis.X


def test_the_hip_sockets_mirror_each_other() -> None:
    landmarks = _pelvis().landmarks
    np.testing.assert_allclose(landmarks["left_hip_socket"].local_position.array, [88, 0, 0])
    np.testing.assert_allclose(landmarks["right_hip_socket"].local_position.array, [-88, 0, 0])


def test_every_sided_pelvis_landmark_has_a_mirrored_partner() -> None:
    landmarks = _pelvis().landmarks
    for name, landmark in landmarks.items():
        if not name.startswith("left_"):
            continue
        partner = landmarks["right_" + name.removeprefix("left_")]
        expected = landmark.local_position.array * [-1.0, 1.0, 1.0]
        np.testing.assert_allclose(partner.local_position.array, expected)


def test_aliases_are_sided_too() -> None:
    resolver = _pelvis().landmark_name_resolver
    assert resolver.resolve(name="left_asis") == "left_anterior_superior_iliac_spine"
    assert resolver.resolve(name="right_psis") == "right_posterior_superior_iliac_spine"
    assert resolver.resolve(name="hips_center") == "pelvis_origin"


def test_the_pelvis_frame_solves_from_its_own_rest_positions() -> None:
    # Feeding the segment its landmarks' rest positions is the sanity check that the
    # frame definition and the authored coordinates agree about which way is which.
    pelvis = _pelvis()
    basis = pelvis.segments["pelvis"].calculate_basis(
        points={name: landmark.local_position for name, landmark in pelvis.landmarks.items()}
    )
    # x runs origin -> left hip socket, which is exactly +x in the authored coordinates.
    np.testing.assert_allclose(basis.x_axis.array, [1.0, 0.0, 0.0], atol=1e-12)
    # y is `sacrum_top` orthogonalized against x. The sacrum sits up AND back, so the
    # resulting y tilts posteriorly - the pelvis frame is not the frame the coordinates
    # are authored in. Reconciling the two is a rest-pose question, not a loader one.
    assert basis.y_axis.array[0] == pytest.approx(0.0, abs=1e-12)
    assert basis.y_axis.array[1] > 0.0
    assert basis.y_axis.array[2] < 0.0


# ── skeleton-level cross validation ───────────────────────────────────


def test_a_landmark_naming_a_nonexistent_segment_is_rejected() -> None:
    component = yaml.safe_load(
        """
        segments:
          PELVIS:
            reference_geometry:
              origin: PELVIS_ORIGIN
              x_axis: {landmark: HIP, type: exact}
        landmarks:
          PELVIS_ORIGIN:
            definition: "origin"
            reference_frame: PELVIS
            local_position: [0, 0, 0]
          HIP:
            definition: "hip"
            reference_frame: PELVIS
            local_position: [88, 0, 0]
          STRAY:
            definition: "belongs to nothing"
            reference_frame: NOT_A_SEGMENT
            local_position: [0, 10, 0]
        """
    )
    landmarks, segments = build_component(component=component, name="pelvis")
    with pytest.raises(ValueError, match="not a segment of this skeleton"):
        SkeletonDefinition(name="pelvis", landmarks=landmarks, segments=segments)


def test_a_segment_with_no_reference_geometry_is_rejected() -> None:
    component = yaml.safe_load("segments:\n  UPPER_ARM:\nlandmarks: {}\n")
    with pytest.raises(ValueError, match="no `reference_geometry`"):
        build_component(component=component, name="arm")

def test_a_component_can_share_a_joint_between_two_segments() -> None:
    # The elbow is the upper arm's distal joint AND the lower arm's origin - one
    # landmark, owned by one segment, referenced by the other as its origin.
    component = yaml.safe_load(
        """
        sided: true
        segments:
          UPPER_ARM:
            reference_geometry:
              origin: SHOULDER
              y_axis: {landmark: ELBOW, type: exact}
          LOWER_ARM:
            reference_geometry:
              origin: ELBOW
              y_axis: {landmark: WRIST, type: exact}
        landmarks:
          SHOULDER:
            definition: "shoulder joint center"
            reference_frame: UPPER_ARM
            local_position: [0, 0, 0]
          ELBOW:
            definition: "elbow joint center"
            reference_frame: UPPER_ARM
            local_position: [0, 300, 0]
          WRIST:
            definition: "wrist joint center"
            reference_frame: LOWER_ARM
            local_position: [0, 260, 0]
        """
    )
    landmarks, segments = build_component(component=component, name="arm")
    skeleton = SkeletonDefinition(name="arm", landmarks=landmarks, segments=segments)
    assert sorted(skeleton.segments) == [
        "left_lower_arm",
        "left_upper_arm",
        "right_lower_arm",
        "right_upper_arm",
    ]
    lower = skeleton.segments["left_lower_arm"]
    assert lower.frame_definition.origin_point_name == "left_elbow"
    assert lower.frame_definition.primary_point_name == "left_wrist"
    assert lower.length == pytest.approx(260.0)
    assert "left_elbow" in lower.landmark_names

def test_the_shipped_hand_yaml_loads() -> None:
    hand = SkeletonDefinition.from_component_yaml(path=HAND_YAML_PATH, name="hand")
    # 20 authored segments x 2 sides, 33 authored landmarks x 2 sides.
    assert len(hand.segments) == 40
    assert len(hand.landmarks) == 66
    # The CMC joint is owned by the carpals and shared as the metacarpal's origin.
    assert hand.segments["left_index_metacarpal"].frame_definition.origin_point_name == "left_index_cmc"
    # The pinky sits on the ulnar side: negative x is legitimate for a fan-shaped hand.
    assert hand.landmarks["left_pinky_cmc"].local_position.array[0] < 0.0
    assert hand.landmarks["right_pinky_cmc"].local_position.array[0] > 0.0

def test_the_shipped_leg_yaml_loads() -> None:
    leg = SkeletonDefinition.from_component_yaml(path=LEG_YAML_PATH, name="leg")
    assert len(leg.segments) == 4
    assert len(leg.landmarks) == 6
    # The knee is owned by the upper leg and shared as the lower leg's origin.
    assert leg.segments["left_lower_leg"].frame_definition.origin_point_name == "left_knee"


def test_the_shipped_foot_yaml_loads() -> None:
    foot = SkeletonDefinition.from_component_yaml(path=FOOT_YAML_PATH, name="foot")
    assert len(foot.segments) == 6
    assert len(foot.landmarks) == 8
    # The ankle is the shared origin of heel and foot; the ball is shared by foot and toes.
    assert foot.segments["left_heel"].frame_definition.origin_point_name == "left_ankle_origin"
    assert foot.segments["left_toes"].frame_definition.origin_point_name == "left_ball"


def test_the_whole_human_skeleton_loads() -> None:
    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    assert skeleton.name == "human"
    assert len(skeleton.segments) == 61
    assert len(skeleton.landmarks) == 169
    # Pelvis, chest, and skull are fully specified; the rest are still roll-underspecified.
    assert set(skeleton.underspecified_segment_names)
    assert "pelvis" not in skeleton.underspecified_segment_names
    assert "chest" not in skeleton.underspecified_segment_names
    assert "skull" not in skeleton.underspecified_segment_names
