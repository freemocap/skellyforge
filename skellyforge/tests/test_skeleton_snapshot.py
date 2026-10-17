"""Saved definitions preserve the rules used to reconstruct segment poses."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_segment
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_snapshot import (
    SkeletonSnapshot,
    SegmentSnapshot,
)


@pytest.mark.parametrize("name", ["pelvis", "thoracic"])
def test_snapshot_preserves_observation_frame_and_reconstructed_pose(name):
    original = SkeletonDefinition.from_default_yaml()
    restored = SkeletonSnapshot.capture(original).restore()
    assert (
        restored.segments[name].observation_frame
        == original.segments[name].observation_frame
    )
    points = {
        n: Point.from_array(values=np.array(p, dtype=float))
        for n, p in {
            "pelvis_origin": [0, 0, 0],
            "left_hip_socket": [-150, 10, 0],
            "right_hip_socket": [150, -10, 0],
            "chest_center": [20, -15, 300],
            "neck_center": [-10, 30, 600],
            "left_acromion": [-180, 15, 675],
            "right_acromion": [160, 45, 525],
        }.items()
    }
    before = hydrate_segment(segment=original.segments[name], observed=points)
    after = hydrate_segment(segment=restored.segments[name], observed=points)
    assert after.solved_by == before.solved_by
    assert after.orientation.is_same_rotation(other=before.orientation)
    np.testing.assert_array_equal(after.origin.array, before.origin.array)
    assert after.scale_estimate == before.scale_estimate


def test_observation_frame_change_is_visible_in_serialized_model():
    snapshot = SkeletonSnapshot.capture(SkeletonDefinition.from_default_yaml())
    changed = replace(
        snapshot,
        segments=tuple(
            replace(s, observation_frame=None) if s.name == "thoracic" else s
            for s in snapshot.segments
        ),
    )
    assert json.dumps(
        asdict(snapshot), sort_keys=True, default=lambda enum: enum.value
    ) != json.dumps(asdict(changed), sort_keys=True, default=lambda enum: enum.value)


def test_legacy_segment_without_observation_frame_does_not_invent_one():
    snapshot = SkeletonSnapshot.capture(SkeletonDefinition.from_default_yaml())
    legacy = []
    for segment in snapshot.segments:
        fields = {
            name: getattr(segment, name)
            for name in SegmentSnapshot.__dataclass_fields__
            if name != "observation_frame"
        }
        legacy.append(SegmentSnapshot(**fields))
    restored = replace(snapshot, segments=tuple(legacy)).restore()
    assert all(s.observation_frame is None for s in restored.segments.values())
    assert (
        restored.segments["thoracic"].frame_definition
        == snapshot.restore().segments["thoracic"].frame_definition
    )


import json
from dataclasses import asdict


def test_json_snapshot_decoder_preserves_saved_definitions_and_legacy_absence():
    snapshot = SkeletonSnapshot.capture(SkeletonDefinition.from_default_yaml())
    data = json.loads(json.dumps(asdict(snapshot), default=lambda item: item.value))
    decoded = SkeletonSnapshot.from_dict(data)
    restored = SkeletonSnapshot.capture(decoded.restore())
    assert (
        json.loads(json.dumps(asdict(restored), default=lambda item: item.value))
        == data
    )
    for segment in data["segments"]:
        segment.pop("observation_frame", None)
    legacy = SkeletonSnapshot.from_dict(data).restore()
    assert all(s.observation_frame is None for s in legacy.segments.values())
