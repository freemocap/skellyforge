"""The tracker→standard-human mapping completeness contract.

The standard human declares ``required_keypoints()`` (76 names). Every
tracker family (rtmpose / mediapipe, each a body + hand mapping pair) must
produce the full required set, verified live at load time by ``tracker_contract``
— the replacement for the deleted golden-fixture snapshot.
"""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

import pytest

from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)
from skellyforge.skellymodels.standard_human import tracker_contract
from skellyforge.skellymodels.standard_human.standard_human_model import (
    compose_standard_human,
)
from skellyforge.skellymodels.standard_human.tracker_contract import (
    validate_all_tracker_families,
    validate_mapping_completeness,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping


def test_all_four_mappings_cover_the_required_set() -> None:
    """The four live mapping YAMLs produce every required keypoint.

    This is the old golden-fixture contract, now checked against the parsed
    mappings (not a stale snapshot) at load time.
    """
    validate_all_tracker_families(compose_standard_human())


def test_gap_raises_with_the_missing_names() -> None:
    """A family that cannot produce every required name fails loudly.

    The ValueError must name the missing required keypoints (sorted).
    """
    human = compose_standard_human()
    body_mapping = TrackerMapping(
        entries={"nose": "nose", "left_shoulder": "left_shoulder"}
    )
    hand_mapping = TrackerMapping(entries={})

    with pytest.raises(ValueError) as exc_info:
        validate_mapping_completeness(human, body_mapping, hand_mapping)

    message = str(exc_info.value)
    assert "left_elbow" in message
    assert "hips" in message


def test_validate_all_names_the_failing_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load-time gap must say which family failed.

    ``validate_mapping_completeness`` reports the missing names; the
    ``validate_all_tracker_families`` wrapper must prefix the family name so
    the error is actionable at load time.  We point the rtmpose body path at an
    empty mapping to force a gap (mediapipe, validated second, never runs).
    """
    human = compose_standard_human()

    def _empty_mapping(_path: Path) -> TrackerMapping:
        return TrackerMapping(entries={})

    monkeypatch.setattr(tracker_contract.TrackerMapping, "from_yaml", _empty_mapping)

    with pytest.raises(ValueError) as exc_info:
        validate_all_tracker_families(human)

    message = str(exc_info.value)
    assert "[rtmpose]" in message
    # The missing-names detail is preserved underneath the prefix.
    assert "does not produce every required standard-human keypoint" in message


def test_hand_names_instantiated_under_both_sides() -> None:
    """Hand names must be unioned under left_/right_, and nothing else.

    A body-only body mapping plus a hand mapping producing just ``wrist`` must
    still raise (the finger names are missing), but must NOT complain about
    ``left_wrist`` / ``right_wrist`` — proving the hand union covered them.
    """
    human = compose_standard_human()
    body_mapping = TrackerMapping(entries={"nose": "nose"})
    hand_mapping = TrackerMapping(entries={"wrist": "wrist"})

    with pytest.raises(ValueError) as exc_info:
        validate_mapping_completeness(human, body_mapping, hand_mapping)

    message = str(exc_info.value)
    # The hand wrist is produced under both sides — no complaint about either.
    assert "left_wrist" not in message
    assert "right_wrist" not in message
    # A finger keypoint (side-instantiated) is still missing.
    assert "left_index_finger_mcp" in message
    assert "right_index_finger_mcp" in message


def test_contract_module_imports_light() -> None:
    """Importing tracker_contract must not pull mediapipe / onnxruntime.

    Ran in a fresh interpreter so heavy deps imported by other tests in this
    session cannot pollute ``sys.modules``.
    """
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import skellyforge.skellymodels.standard_human.tracker_contract; "
            "print('mediapipe' in sys.modules, 'onnxruntime' in sys.modules)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"
