"""The tracker→standard-human mapping completeness contract.

The standard human declares ``required_keypoints()`` — every keypoint its
segments are driven by.  Each tracker family (rtmpose / mediapipe) ships a
body + hand mapping YAML; every required keypoint must be produced by one of
the two, or the model fails loudly at load time.  This *is* the one sanctioned
lateral import in skellyforge: the sub-skelly boundary is deliberately leaky
here, because maintaining it would mean duplicating skellytracker's mapping
machinery or shipping a stale golden fixture.  Only skellytracker's ``core.io``
modules (base install, no detector machinery) are imported, and only here —
skellyforge's ``__init__.py`` must NOT import this module.
"""

from __future__ import annotations

from pathlib import Path

from skellyforge.skellymodels.standard_human.standard_human_model import StandardHuman
from skellytracker.core.io.mapping_paths import (
    MEDIAPIPE_BODY_MAPPING,
    MEDIAPIPE_HAND_MAPPING,
    RTMPOSE_BODY_MAPPING,
    RTMPOSE_HAND_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping


def _produced_keypoints(
    body_mapping: TrackerMapping,
    hand_mapping: TrackerMapping,
) -> set[str]:
    """The standard-human keypoint set a tracker family produces.

    The hand mapping is authored side-agnostically; each hand keypoint name is
    instantiated under both ``left_`` and ``right_`` (mirroring how
    ``compose_parts`` expands ``HAND_PART``), then unioned with the body names.
    """
    produced: set[str] = set(body_mapping.keypoint_names)
    for name in hand_mapping.keypoint_names:
        produced.add(f"left_{name}")
        produced.add(f"right_{name}")
    return produced


def validate_mapping_completeness(
    human: StandardHuman,
    body_mapping: TrackerMapping,
    hand_mapping: TrackerMapping,
) -> None:
    """Fail loudly if a tracker family omits any required keypoint.

    Parameters
    ----------
    human
        The composed standard human whose ``required_keypoints()`` is the
        contract set.
    body_mapping
        The family's body mapping (tracker body keypoints → standard-human).
    hand_mapping
        The family's hand mapping (side-agnostic; instantiated under both
        sides).

    Raises
    ------
    ValueError
        Listing every required keypoint the family does not produce (sorted),
        with a clear message.  A gap is a load-time error, not a warning.
    """
    missing = sorted(human.required_keypoints() - _produced_keypoints(body_mapping, hand_mapping))
    if missing:
        raise ValueError(
            "tracker mapping does not produce every required standard-human "
            "keypoint; missing (" + ", ".join(missing) + ")"
        )


def validate_all_tracker_families(human: StandardHuman) -> None:
    """Validate every tracker family against the required keypoint set.

    Loads the four mappings from the registry paths and checks each of the two
    families (rtmpose body+hand, mediapipe body+hand).  Raises on the first gap.
    """
    families: tuple[tuple[str, Path, Path], ...] = (
        ("rtmpose", RTMPOSE_BODY_MAPPING, RTMPOSE_HAND_MAPPING),
        ("mediapipe", MEDIAPIPE_BODY_MAPPING, MEDIAPIPE_HAND_MAPPING),
    )
    for family, body_path, hand_path in families:
        body_mapping = TrackerMapping.from_yaml(body_path)
        hand_mapping = TrackerMapping.from_yaml(hand_path)
        try:
            validate_mapping_completeness(human, body_mapping, hand_mapping)
        except ValueError as exc:
            raise ValueError(f"[{family}] {exc}") from exc
