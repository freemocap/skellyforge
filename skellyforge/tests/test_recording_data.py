"""Recording-reader regressions retained from the diagnostic suite."""
import numpy as np
import pytest
from skellyforge.tools.recording_data import attach_keypoints, decode_rows, recording_path
from skellyforge.tools.recording_data import decode_batches


@pytest.mark.parametrize("bad", [None, "duplicate", "units", "time", "component", "nan", "null"])
def test_column_batches_match_row_reader_and_reject_invalid_data(bad):
    pa = pytest.importorskip("pyarrow")
    samples = rows()
    if bad == "duplicate":
        samples.append(samples[0])
    elif bad == "units":
        samples[0]["units"] = "px"
    elif bad == "time":
        samples[0]["timestamp_s"] = 99.
    elif bad == "component":
        samples[0]["component"] = "w"
    elif bad == "nan":
        samples[0]["value"] = np.nan
    elif bad == "null":
        samples[0]["value"] = None
    batches = pa.Table.from_pylist(samples).to_batches(max_chunksize=2)
    if bad not in (None, "null"):
        with pytest.raises(ValueError):
            decode_batches(batches, ["point"])
    else:
        actual = decode_batches(batches, ["point"])
        expected = decode_rows(samples, ["point"])
        for a, b in zip(actual, expected, strict=True):
            assert a["number"] == b["number"] and a["time"] == b["time"]
            assert list(a["points"]) == list(b["points"])
            for n in a["points"]:
                np.testing.assert_array_equal(a["points"][n], b["points"][n])


def test_column_reader_preserves_first_seen_point_order_and_shuffled_rows():
    pa = pytest.importorskip("pyarrow")
    samples = [dict(r, name=name) for name in ("b", "a") for r in rows()]
    samples = [samples[i] for i in np.random.default_rng(83).permutation(len(samples))]
    actual = decode_batches(pa.Table.from_pylist(samples).to_batches(max_chunksize=3), ["a", "b"])
    expected = decode_rows(samples, ["a", "b"])
    for a, b in zip(actual, expected, strict=True):
        assert list(a["points"]) == list(b["points"])
        for n in a["points"]:
            np.testing.assert_array_equal(a["points"][n], b["points"][n])


def test_column_reader_empty_and_all_null_channels():
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="nonempty"):
        decode_batches([], ["point"])
    table = pa.Table.from_pylist([dict(r, value=None) for r in rows()])
    result = decode_batches(table.to_batches(), ["point"])
    assert len(result) == 2 and all(not frame["points"] for frame in result)


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


def test_prefers_prepared_recording_without_creating_or_downloading_data(tmp_path_factory):
    # The nested recording layout needs a short fixture name on Windows.
    tmp_path = tmp_path_factory.mktemp("r")
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
