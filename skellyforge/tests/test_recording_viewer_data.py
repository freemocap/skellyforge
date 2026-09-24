"""Prepared data discovery and scalar-to-landmark reconstruction contracts."""

import numpy as np
import pytest
from types import SimpleNamespace

from scripts.recording_data import attach_keypoints, decode_rows, recording_path


def saved_fixture():
    # Deliberately unrelated to the default human skeleton and landmark positions.
    return (
        {
            "origins": {"bone": np.array([10.0, 20.0, 30.0])},
            "rotations": {"bone": np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])},
        },
        {
            "landmarks": [{"name": "tip", "position": [2.0, 0.0, 0.0]}],
            "segments": [{"name": "bone", "frame": {"primary_point": "tip"}}],
        },
        SimpleNamespace(segment_scales={"bone": 3.0}, segment_lengths={"bone": 6.0}),
    )


def test_drawing_uses_saved_transform_geometry_and_fixed_scale():
    from scripts.generate_real_skeleton_viewer import saved_segments

    record, skeleton, fit = saved_fixture()
    result = saved_segments(record, skeleton, fit)["bone"]
    np.testing.assert_array_equal(result["origin"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(result["end"], [10.0, 26.0, 30.0], atol=1e-12)
    np.testing.assert_array_equal(
        result["quaternion_wxyz"], record["rotations"]["bone"]
    )
    np.testing.assert_allclose(result["axes"][0], [0.0, 1.0, 0.0], atol=1e-12)


def test_missing_saved_pose_is_not_reconstructed():
    from scripts.generate_real_skeleton_viewer import saved_segments

    record, skeleton, fit = saved_fixture()
    record["rotations"] = {}
    assert saved_segments(record, skeleton, fit) == {}


@pytest.mark.parametrize("bad", ["quaternion", "length"])
def test_saved_pose_inconsistency_fails_without_repair(bad):
    from scripts.generate_real_skeleton_viewer import saved_segments

    record, skeleton, fit = saved_fixture()
    if bad == "quaternion":
        record["rotations"]["bone"] *= 2
    else:
        fit.segment_lengths["bone"] = 8.0
    with pytest.raises(ValueError):
        saved_segments(record, skeleton, fit)


def test_quaternion_channel_preserves_wxyz_and_missing_values():
    samples = [
        dict(
            frame_number=0,
            timestamp_s=0.0,
            name="bone",
            component=c,
            value=v,
            units="1",
        )
        for c, v in zip("wxyz", [1.0, 0.0, 0.0, 0.0])
    ]
    result = decode_rows(samples, ["bone"], "wxyz", "1")
    np.testing.assert_array_equal(result[0]["points"]["bone"], [1.0, 0.0, 0.0, 0.0])
    samples[0]["value"] = None
    assert decode_rows(samples, ["bone"], "wxyz", "1")[0]["points"] == {}


def test_keypoints_remain_distinct_from_same_named_landmarks():
    landmarks = decode_rows(rows(), ["point"])
    samples = rows()
    samples[0]["value"] = 7.0
    keypoints = decode_rows(samples, ["point"])
    attach_keypoints(landmarks, keypoints)
    assert landmarks[0]["points"]["point"][0] == 1.0
    assert landmarks[0]["keypoints"]["point"][0] == 7.0


@pytest.mark.parametrize("field,value", [("number", 3), ("time", 1.3)])
def test_keypoint_overlay_rejects_misaligned_frames(field, value):
    landmarks = decode_rows(rows(), ["point"])
    keypoints = decode_rows(rows(), ["point"])
    keypoints[0][field] = value
    with pytest.raises(ValueError, match="identical frame"):
        attach_keypoints(landmarks, keypoints)


def rows():
    return [
        dict(
            frame_number=f,
            timestamp_s=t,
            name="point",
            component=c,
            value=v,
            units="mm",
        )
        for f, t in [(2, 1.2), (5, 1.9)]
        for c, v in zip("xyz", [1.0, 2.0, 3.0])
    ]


def test_preserves_frame_numbers_timestamps_and_omits_incomplete_points():
    samples = rows()
    samples[-1]["value"] = None
    result = decode_rows(samples, ["point"])
    assert [f["number"] for f in result] == [2, 5]
    assert [f["time"] for f in result] == [1.2, 1.9]
    np.testing.assert_array_equal(result[0]["points"]["point"], [1, 2, 3])
    assert result[1]["points"] == {}


@pytest.mark.parametrize("bad", ["duplicate", "units", "time", "component"])
def test_rejects_ambiguous_or_incompatible_samples(bad):
    samples = rows()
    if bad == "duplicate":
        samples.append(samples[0])
    if bad == "units":
        samples[0]["units"] = "px"
    if bad == "time":
        samples[0]["timestamp_s"] = 2.0
    if bad == "component":
        samples[0]["component"] = "w"
    with pytest.raises(ValueError):
        decode_rows(samples, ["point"])


def test_prefers_prepared_recording_without_creating_or_downloading_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepare it in FreeMoCap"):
        recording_path(home=tmp_path)
    name = "freemocap_test_data"
    canonical = (
        tmp_path / "freemocap_data" / "recordings" / name / f"{name}_data.parquet"
    )
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    assert recording_path(home=tmp_path) == canonical
    prepared = (
        tmp_path
        / "freemocap_data"
        / "testing"
        / "prepared"
        / name
        / "current"
        / "recordings"
        / name
        / f"{name}_data.parquet"
    )
    prepared.parent.mkdir(parents=True)
    prepared.touch()
    assert recording_path(home=tmp_path) == prepared
