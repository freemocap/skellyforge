"""Scientific NumPy archives preserve declared content and reject ambiguous input."""
from dataclasses import replace
import gc
import json
import weakref
from zipfile import ZipFile, ZipInfo, ZIP_STORED, ZIP_DEFLATED

import numpy as np
import pytest

from skellyforge.core.data_tensors import (
    AxisDefinition, CompanionTensor, DataTensor, DefinitionDocument, NumpyDatasetWriter,
    RecordingDataset, SampleTiming, ScientificProvenance, SourceIdentity, TensorLayout,
    TimeSeries, read_numpy_dataset, write_numpy_dataset,
)


def scientific_dataset():
    source = SourceIdentity("model:fixture", instance_id=2, sensor_group="group", model_id="fixture")
    timing = SampleTiming(timeline_id="camera_capture", clock_id="camera_clock", source_identity=source,
                          sample_ids=[4, 9], timestamps_s=[-0., .25], clock_epoch=3)
    axes = (AxisDefinition("entity_identity", ("second", "first")),
            AxisDefinition("capture_samples", timing.sample_ids),
            AxisDefinition("component_identity", ("a", "b", "c", "d", "e")))
    values = np.arange(40, dtype=">f8").reshape(2, 2, 10)[..., ::2]
    values[0, 0, 0], values[1, 1, 4] = -0., np.nan
    layout = TensorLayout(axes, values.dtype, ("mm", "mm", "mm", "rad", "1"),
                          reference_id="declared_reference", source_identity=source, channel_kind="positions")
    presence = np.ones(values.shape, dtype=bool)
    presence[1, 1, 4] = False
    companions = (
        CompanionTensor("entry_presence", presence, TensorLayout(axes, bool)),
        CompanionTensor("value_validity", np.isfinite(values), TensorLayout(axes, bool)),
        CompanionTensor("confidence_scores", np.array([[.3, .4], [.5, .6]], dtype=np.float32), TensorLayout(axes[:2], np.float32)),
        CompanionTensor("measured_support", np.array([True, False]), TensorLayout(axes[:1], bool)),
    )
    provenance = ScientificProvenance("model_prediction", "fixture_operation", ("input_a",), "fixture")
    first = TimeSeries(DataTensor(values, layout, provenance, companions), timing, "capture_samples")
    complex_layout = TensorLayout((timing.sample_axis,), "complex128", source_identity=source, channel_kind="complex_signal")
    shared = TimeSeries(DataTensor(np.array([1 + 2j, -3 + 4j]), complex_layout, provenance), timing)
    independent_timing = replace(timing, timeline_id="sensor_capture", clock_id="sensor_clock")
    independent = TimeSeries(DataTensor(np.array([3, 9], dtype=np.int16),
        replace(complex_layout, dtype=np.int16, channel_kind="sensor_counts"), provenance), independent_timing)
    scalar = DataTensor(np.asarray(-0.), TensorLayout((), "float64", source_identity=source, channel_kind="static_scalar"), provenance)
    segment_axis = AxisDefinition("segment_identity", ("arm", "leg"))
    static_layout = TensorLayout((segment_axis,), "float64", source_identity=source, channel_kind="segment_lengths")
    static = DataTensor(np.array([3., 5.]), static_layout, provenance,
        (CompanionTensor("voting_support", np.array([False, True]), TensorLayout((segment_axis,), bool)),))
    return RecordingDataset("fixture_recording", 7,
        tuple(series.immutable_copy() for series in (first, shared, independent)),
        tuple(tensor.immutable_copy() for tensor in (scalar, static)),
        (DefinitionDocument("model:fixture", '{ "ordered": ["second", "first"] }'),))


def assert_tensor_equal(actual, expected):
    assert actual.tensor_layout == expected.tensor_layout
    assert actual.scientific_provenance == expected.scientific_provenance
    assert actual.data_values.dtype.str == expected.data_values.dtype.str
    np.testing.assert_array_equal(actual.data_values, expected.data_values)
    if expected.data_values.dtype.kind == "f":
        np.testing.assert_array_equal(np.signbit(actual.data_values), np.signbit(expected.data_values))
    assert actual.is_immutable
    with pytest.raises(ValueError):
        actual.data_values.flags.writeable = True
    assert len(actual.companion_tensors) == len(expected.companion_tensors)
    for found, wanted in zip(actual.companion_tensors, expected.companion_tensors, strict=True):
        assert found.metadata_name == wanted.metadata_name
        assert found.tensor_layout == wanted.tensor_layout
        np.testing.assert_array_equal(found.data_values, wanted.data_values)
        with pytest.raises(ValueError):
            found.data_values.flags.writeable = True


@pytest.mark.parametrize("format", ["npy", "npz"])
def test_roundtrip_scientific_content_and_plain_numpy(tmp_path, format):
    expected = scientific_dataset()
    path = tmp_path / ("dataset" if format == "npy" else "dataset.npz")
    manifest = expected.save_numpy(path, format=format)
    actual = RecordingDataset.load_numpy(path)
    assert actual.recording_id == expected.recording_id
    assert actual.run_id == expected.run_id
    assert actual.definition_documents == expected.definition_documents
    for found, wanted in zip(actual.time_series, expected.time_series, strict=True):
        assert_tensor_equal(found.data_tensor, wanted.data_tensor)
        assert found.sample_axis_name == wanted.sample_axis_name
        np.testing.assert_array_equal(found.sample_timing.sample_ids, wanted.sample_timing.sample_ids)
        np.testing.assert_array_equal(found.sample_timing.timestamps_s, wanted.sample_timing.timestamps_s)
        np.testing.assert_array_equal(np.signbit(found.sample_timing.timestamps_s), np.signbit(wanted.sample_timing.timestamps_s))
        assert found.sample_timing.clock_id == wanted.sample_timing.clock_id
        assert found.sample_timing.clock_epoch == wanted.sample_timing.clock_epoch
        assert found.sample_timing.source_identity == wanted.sample_timing.source_identity
    assert actual.time_series[0].sample_timing is actual.time_series[1].sample_timing
    assert actual.time_series[2].sample_timing is not actual.time_series[0].sample_timing
    assert len(actual.sample_timings) == 2
    for found, wanted in zip(actual.static_tensors, expected.static_tensors, strict=True):
        assert_tensor_equal(found, wanted)
    channel = manifest["channels"][0]
    assert channel["tensor_layout"]["axes"][1] == {"axis_name": "capture_samples", "timing_ref": "timing_000000"}
    if format == "npy":
        parsed = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
        array = np.load(path / parsed["channels"][0]["values_file"], allow_pickle=False, mmap_mode="r")
        np.testing.assert_array_equal(array, expected.time_series[0].data_tensor.data_values)
        array._mmap.close()
        (path / "README.md").write_text("Optional guide", encoding="utf-8")
        read_numpy_dataset(path)
    else:
        with ZipFile(path) as archive:
            assert all(info.compress_type == ZIP_DEFLATED for info in archive.infolist())
            parsed = json.loads(archive.read("dataset.json"))
            with archive.open(parsed["channels"][0]["values_file"]) as stream:
                array = np.load(stream, allow_pickle=False)
            np.testing.assert_array_equal(array, expected.time_series[0].data_tensor.data_values)


@pytest.mark.parametrize("format", ["npy", "npz"])
def test_empty_dataset_and_empty_timing(tmp_path, format):
    empty = RecordingDataset("empty_recording", 0)
    path = tmp_path / "empty"
    write_numpy_dataset(empty, path, format=format, compressed=False)
    assert read_numpy_dataset(path).channel_identities == ()
    if format == "npz":
        with ZipFile(path) as archive:
            assert archive.getinfo("dataset.json").compress_type == ZIP_STORED
    source = scientific_dataset().time_series[0]
    empty_series = source.slice_samples(1, 1)
    path = tmp_path / "empty_series"
    write_numpy_dataset(RecordingDataset("empty_series", 0, (empty_series,)), path, format=format)
    restored = read_numpy_dataset(path)
    assert restored.time_series[0].sample_count == 0
    assert_tensor_equal(restored.time_series[0].data_tensor, empty_series.data_tensor)


def test_streaming_borrows_noncontiguous_arrays_then_releases_them(tmp_path):
    source = scientific_dataset().time_series[0]
    values = np.arange(40, dtype=">f8").reshape(2, 2, 10)[..., ::2]
    reference = weakref.ref(values)
    tensor = DataTensor(values, source.data_tensor.tensor_layout, source.data_tensor.scientific_provenance)
    series = TimeSeries(tensor, source.sample_timing, source.sample_axis_name)
    with NumpyDatasetWriter(tmp_path / "streamed", recording_id="streamed", run_id=0) as writer:
        writer.add_static(scientific_dataset().static_tensors[0])
        entry = writer.add_series(series)
        entry["tensor_layout"]["axes"].clear()
        del series, tensor, values
        gc.collect()
        assert reference() is None
    assert writer.manifest["channels"][0]["storage_kind"] == "time_series"
    assert set(writer.filenames) == {item.name for item in (tmp_path / "streamed").iterdir()}
    restored = read_numpy_dataset(tmp_path / "streamed")
    np.testing.assert_array_equal(restored.time_series[0].data_tensor.data_values,
                                  np.arange(40).reshape(2, 2, 10)[..., ::2])


@pytest.mark.parametrize("format", ["npy", "npz"])
def test_write_failure_cleans_only_owned_destination(tmp_path, monkeypatch, format):
    dataset = scientific_dataset()
    target = tmp_path / "failed"
    original = NumpyDatasetWriter._write_array
    def fail_after_write(self, filename, values):
        original(self, filename, values)
        raise RuntimeError("injected write failure")
    monkeypatch.setattr(NumpyDatasetWriter, "_write_array", fail_after_write)
    with pytest.raises(RuntimeError, match="injected"):
        write_numpy_dataset(dataset, target, format=format)
    assert not target.exists()
    target.write_text("existing user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_numpy_dataset(dataset, target, format=format)
    assert target.read_text(encoding="utf-8") == "existing user data"


def test_abort_does_not_remove_unowned_files_in_destination(tmp_path):
    path = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional"):
        with NumpyDatasetWriter(path, recording_id="failure", run_id=0) as writer:
            writer.add_static(scientific_dataset().static_tensors[0])
            (path / "user_note.txt").write_text("preserve", encoding="utf-8")
            raise RuntimeError("intentional")
    assert list(path.iterdir()) == [path / "user_note.txt"]


@pytest.mark.parametrize("name,dtype", [("solver_status", np.uint8), ("hydration_status", np.uint8),
                                       ("unknown_quality", float), ("voting_support", float)])
def test_rejects_unsupported_or_mistyped_companions(tmp_path, name, dtype):
    source = scientific_dataset().static_tensors[1]
    companion = CompanionTensor(name, np.ones(source.data_values.shape, dtype=dtype),
                                replace(source.tensor_layout, dtype=dtype))
    tensor = DataTensor(source.data_values, source.tensor_layout, source.scientific_provenance, (companion,))
    with pytest.raises(ValueError, match="companion"):
        with NumpyDatasetWriter(tmp_path / "rejected", recording_id="invalid", run_id=0) as writer:
            writer.add_static(tensor)
    assert not (tmp_path / "rejected").exists()


def test_rejects_static_sample_axis_even_when_static_added_first(tmp_path):
    series = scientific_dataset().time_series[0]
    tensor = DataTensor(np.array([1., 2.]),
        TensorLayout((AxisDefinition("capture_samples", (4, 9)),), "float64",
                     source_identity=SourceIdentity("static_source"), channel_kind="static_values"),
        series.data_tensor.scientific_provenance)
    with pytest.raises(ValueError, match="sample axis"):
        with NumpyDatasetWriter(tmp_path / "conflict", recording_id="invalid", run_id=0) as writer:
            writer.add_static(tensor)
            writer.add_series(series)


def test_shared_timing_rejects_signed_zero_conflict(tmp_path):
    first, second = scientific_dataset().time_series[:2]
    conflicting = replace(second.sample_timing, timestamps_s=[0., .25])
    with pytest.raises(ValueError, match="timing identity"):
        with NumpyDatasetWriter(tmp_path / "conflict", recording_id="invalid", run_id=0) as writer:
            writer.add_series(first)
            writer.add_series(TimeSeries(second.data_tensor, conflicting))


@pytest.mark.parametrize("format", ["npy", "npz"])
@pytest.mark.parametrize("corruption", ["finite_value", "signed_zero", "complex_nan", "complex_zero", "trailing_bytes"])
def test_verification_rejects_corruption_and_cleans_output(tmp_path, monkeypatch, format, corruption):
    if corruption == "complex_nan":
        expected, corrupt = np.asarray(complex(1., np.nan)), np.asarray(complex(2., np.nan))
    elif corruption == "complex_zero":
        expected, corrupt = np.asarray(complex(-0., -0.)), np.asarray(complex(0., -0.))
    elif corruption == "signed_zero":
        expected, corrupt = np.asarray(-0.), np.asarray(0.)
    else:
        expected, corrupt = np.asarray(1.), np.asarray(2.)
    source = scientific_dataset().static_tensors[0]
    tensor = DataTensor(expected, replace(source.tensor_layout, dtype=expected.dtype), source.scientific_provenance)
    original = np.lib.format.write_array
    def write_corrupted(stream, values, **kwargs):
        original(stream, values if corruption == "trailing_bytes" else corrupt, **kwargs)
        if corruption == "trailing_bytes":
            stream.write(b"x")
    monkeypatch.setattr(np.lib.format, "write_array", write_corrupted)
    target = tmp_path / "corrupt"
    with pytest.raises(ValueError, match="verification|trailing bytes"):
        with NumpyDatasetWriter(target, recording_id="failure", run_id=0, format=format) as writer:
            writer.add_static(tensor)
    assert not target.exists()


def test_verification_can_be_explicitly_disabled(tmp_path, monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Verification was disabled")
    monkeypatch.setattr(np.lib.format, "read_array", unexpected_read)
    write_numpy_dataset(scientific_dataset(), tmp_path / "unverified", verify=False)


@pytest.mark.parametrize("dtype", ["bool", "uint64", "float16", ">i4", ">c16"])
@pytest.mark.parametrize("format", ["npy", "npz"])
def test_numeric_dtype_fidelity(tmp_path, dtype, format):
    source = scientific_dataset().static_tensors[0]
    values = np.asarray([0, 2, 5], dtype=dtype)
    layout = TensorLayout((AxisDefinition("component_identity", ("alpha", "beta", "gamma")),),
                          values.dtype, ("1", "1", "1"), source_identity=source.tensor_layout.source_identity,
                          channel_kind="numeric_variants")
    tensor = DataTensor(values, layout, source.scientific_provenance).immutable_copy()
    dataset = RecordingDataset("dtype_fixture", 0, static_tensors=(tensor,))
    path = tmp_path / "dataset"
    write_numpy_dataset(dataset, path, format=format)
    assert_tensor_equal(read_numpy_dataset(path).static_tensors[0], tensor)


def saved_manifest(tmp_path):
    path = tmp_path / "dataset"
    write_numpy_dataset(scientific_dataset(), path)
    return path, json.loads((path / "dataset.json").read_text(encoding="utf-8"))


def test_nonnative_endian_timing_loads_as_native_fixed_width(tmp_path):
    path, manifest = saved_manifest(tmp_path)
    timing = manifest["timings"][0]
    np.save(path / timing["sample_ids_file"], np.asarray([4, 9], dtype=">i8"), allow_pickle=False)
    np.save(path / timing["timestamps_s_file"], np.asarray([-0., .25], dtype=">f8"), allow_pickle=False)
    actual = read_numpy_dataset(path).time_series[0].sample_timing
    assert actual.sample_ids.dtype == np.dtype("int64")
    assert actual.timestamps_s.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual.sample_ids, [4, 9])
    np.testing.assert_array_equal(actual.timestamps_s, [-0., .25])
    assert np.signbit(actual.timestamps_s[0])


@pytest.mark.parametrize("content", ['{"mass":1e999}', '{"mass":1,"mass":2}'])
def test_definition_json_rejects_overflow_and_duplicate_fields(tmp_path, content):
    document = DefinitionDocument("invalid_definition", content)
    with pytest.raises(ValueError):
        NumpyDatasetWriter(tmp_path / "invalid", recording_id="invalid", run_id=0, definition_documents=(document,))
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("mutation", ["version", "version_bool", "extra_field", "unsafe_path", "duplicate_file",
                                       "dtype", "axis", "duplicate_channel", "bad_timing", "unknown_companion"])
def test_rejects_invalid_manifest(tmp_path, mutation):
    path, manifest = saved_manifest(tmp_path)
    channel = manifest["channels"][0]
    if mutation == "version": manifest["version"] = 99
    elif mutation == "version_bool": manifest["version"] = True
    elif mutation == "extra_field": manifest["extra"] = 0
    elif mutation == "unsafe_path": channel["values_file"] = "../outside.npy"
    elif mutation == "duplicate_file": channel["companion_tensors"][0]["values_file"] = channel["values_file"]
    elif mutation == "dtype": channel["tensor_layout"]["dtype"] = "<f4"
    elif mutation == "axis": channel["tensor_layout"]["axes"][0]["identifiers"] = ["first"]
    elif mutation == "duplicate_channel":
        manifest["channels"][1]["tensor_layout"]["channel_kind"] = "positions"
        manifest["channels"][1]["tensor_layout"]["reference_id"] = "declared_reference"
    elif mutation == "bad_timing": channel["timing_ref"] = "absent_timing"
    elif mutation == "unknown_companion": channel["companion_tensors"][0]["metadata_name"] = "solver_status"
    (path / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        read_numpy_dataset(path)


@pytest.mark.parametrize("damage", ["object_array", "wrong_shape", "timing_dtype", "extra_file", "missing_file", "trailing_bytes", "duplicate_key"])
def test_rejects_invalid_payload_and_inventory(tmp_path, damage):
    path, manifest = saved_manifest(tmp_path)
    filename = path / manifest["channels"][0]["values_file"]
    if damage == "object_array": np.save(filename, np.array([{"unsafe": True}], dtype=object), allow_pickle=True)
    elif damage == "wrong_shape": np.save(filename, np.ones((2,), dtype=">f8"), allow_pickle=False)
    elif damage == "timing_dtype": np.save(path / manifest["timings"][0]["sample_ids_file"], np.array([4, 9], dtype=np.int32))
    elif damage == "extra_file": (path / "extra.npy").write_bytes(b"unexpected")
    elif damage == "missing_file": filename.unlink()
    elif damage == "trailing_bytes":
        with filename.open("ab") as stream: stream.write(b"extra")
    elif damage == "duplicate_key":
        text = (path / "dataset.json").read_text(encoding="utf-8")
        (path / "dataset.json").write_text(text.replace('"version":1', '"version":1,"version":1'), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        read_numpy_dataset(path)


@pytest.mark.parametrize("damage", ["duplicate", "link", "extra"])
def test_rejects_invalid_zip_inventory(tmp_path, damage):
    path = tmp_path / "dataset.npz"
    write_numpy_dataset(scientific_dataset(), path, format="npz")
    with ZipFile(path, "a") as archive:
        if damage == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("dataset.json", archive.read("dataset.json"))
        elif damage == "link":
            entry = ZipInfo("extra.npy")
            entry.create_system = 3
            entry.external_attr = 0o120777 << 16
            archive.writestr(entry, "target.npy")
        else: archive.writestr("README.md", "not allowed in NPZ")
    with pytest.raises(ValueError):
        read_numpy_dataset(path)
