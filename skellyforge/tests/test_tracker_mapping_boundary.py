"""The SkellyTracker -> SkellyForge mapping boundary.

SkellyTracker owns the tracker-keypoint -> standard-human-landmark mapping YAMLs;
SkellyForge owns the landmark vocabulary. This test loads SkellyTracker's four mapping
files and asserts every key is a real SkellyForge landmark, so a landmark rename in
SkellyForge fails this test until the SkellyTracker mappings are updated.

The module skips itself when skellytracker is not installed (e.g. a bare uv sync
--no-dev), so the core suite still runs without the dev group.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

pytest.importorskip("skellytracker", reason="skellytracker is a dev dependency")

from skellytracker.core.io.mapping_paths import (  # noqa: E402
    MEDIAPIPE_BODY_MAPPING,
    MEDIAPIPE_HAND_MAPPING,
    RTMPOSE_BODY_MAPPING,
    RTMPOSE_HAND_MAPPING,
)

_SKELETON_YAML = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)

_BODY_MAPPINGS = (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING)
_HAND_MAPPINGS = (MEDIAPIPE_HAND_MAPPING, RTMPOSE_HAND_MAPPING)
_ALL_MAPPINGS = _BODY_MAPPINGS + _HAND_MAPPINGS


def _mapping_keys(path: Path) -> set[str]:
    return set(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_every_mapping_key_is_a_skellyforge_landmark() -> None:
    skeleton = SkeletonDefinition.from_yaml(path=_SKELETON_YAML)
    landmark_names = set(skeleton.landmarks)

    offenders: list[str] = []
    for mapping_path in _ALL_MAPPINGS:
        for key in _mapping_keys(mapping_path):
            if key not in landmark_names:
                offenders.append(f"{mapping_path.name}: {key!r}")

    assert not offenders, (
        "these mapping keys are not SkellyForge landmarks - update the SkellyTracker "
        "mapping to match: " + ", ".join(offenders)
    )


def test_body_and_hand_mappings_do_not_overlap() -> None:
    body = _mapping_keys(MEDIAPIPE_BODY_MAPPING) | _mapping_keys(RTMPOSE_BODY_MAPPING)
    hand = _mapping_keys(MEDIAPIPE_HAND_MAPPING) | _mapping_keys(RTMPOSE_HAND_MAPPING)
    overlap = body & hand
    assert not overlap, f"body and hand mappings overlap on: {sorted(overlap)}"
