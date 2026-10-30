"""Stream scientific tensor channels to portable NPY directories or NPZ archives."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import stat
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

import numpy as np

from skellyforge.core.data_tensors.data_tensor import CompanionTensor, DataTensor
from skellyforge.core.data_tensors.numpy_manifest import (
    FORMAT_NAME, FORMAT_VERSION, definition_data, definitions_from_data, integer_value,
    layout_data, layout_from_data, manifest_inventory, parse_json, provenance_from_data,
    source_data, source_from_data, text_value, validate_scientific_tensor,
)
from skellyforge.core.data_tensors.recording_dataset import ChannelIdentity, RecordingDataset
from skellyforge.core.data_tensors.sample_timing import SampleTiming
from skellyforge.core.data_tensors.time_series import TimeSeries


def _same_timing(first, second):
    return first is second or (
        first.timeline_id == second.timeline_id and first.clock_id == second.clock_id
        and first.clock_epoch == second.clock_epoch and first.source_identity == second.source_identity
        and np.array_equal(first.sample_ids, second.sample_ids)
        and np.array_equal(first.timestamps_s.view(np.uint64), second.timestamps_s.view(np.uint64))
    )


def _verify_array_values(expected, actual):
    if expected.dtype.str != actual.dtype.str or expected.shape != actual.shape:
        raise ValueError("Written array dtype or shape failed verification")
    parts = ((expected.real, actual.real), (expected.imag, actual.imag)) if expected.dtype.kind == "c" else ((expected, actual),)
    for wanted, found in parts:
        if not np.array_equal(wanted, found, equal_nan=True):
            raise ValueError("Written array values failed verification")
        if wanted.dtype.kind == "f":
            zeros = wanted == 0
            if not np.array_equal(np.signbit(wanted[zeros]), np.signbit(found[zeros])):
                raise ValueError("Written array signed zeros failed verification")


class NumpyDatasetWriter:
    """Write borrowed channels immediately; retain metadata and shared timing only."""

    def __init__(self, path, *, recording_id, run_id, definition_documents=(), format="npy", compressed=True, verify=True):
        if format not in ("npy", "npz") or type(compressed) is not bool or type(verify) is not bool:
            raise ValueError("Expected npy/npz format and boolean compression/verification choices")
        self.path = Path(path).absolute()
        self.format = format
        self.compressed = compressed
        self.verify = verify
        self._manifest = dict(format=FORMAT_NAME, version=FORMAT_VERSION,
            recording_id=text_value(recording_id), run_id=integer_value(run_id),
            definition_documents=definition_data(definition_documents), timings=[], channels=[])
        self._timings = {}
        self._channel_keys = set()
        self._sample_axes = {"sample_identity"}
        self._static_axes = set()
        self._filenames = set()
        self._created_files = []
        self._archive = None
        self._archive_file = None
        self._state = "new"

    @property
    def manifest(self):
        """Return detached metadata, including after the writer closes."""
        return deepcopy(self._manifest)

    @property
    def filenames(self):
        return tuple(sorted(self._filenames))

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("Dataset writer cannot be reopened")
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        if self.format == "npy":
            self.path.mkdir()
        else:
            self._archive_file = self.path.open("x+b")
        self._state = "open"
        try:
            if self.format == "npz":
                self._archive = ZipFile(self._archive_file, "w",
                    compression=ZIP_DEFLATED if self.compressed else ZIP_STORED, allowZip64=True)
        except BaseException:
            self._abort()
            raise
        return self

    def _require_open(self):
        if self._state != "open":
            raise RuntimeError("Dataset writer is not open")

    def _write_array(self, filename, values):
        if filename in self._filenames:
            raise ValueError("Duplicate output filename")
        if self.format == "npy":
            target = self.path / filename
            with target.open("x+b") as stream:
                self._created_files.append(target)
                np.lib.format.write_array(stream, values, allow_pickle=False)
                if self.verify:
                    stream.flush()
                    stream.seek(0)
                    self._verify_written_array(stream, values)
        else:
            with self._archive.open(filename, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, values, allow_pickle=False)
            if self.verify:
                with self._archive.open(filename) as stream:
                    self._verify_written_array(stream, values)
        self._filenames.add(filename)

    @staticmethod
    def _verify_written_array(stream, values):
        actual = np.lib.format.read_array(stream, allow_pickle=False)
        if stream.read(1):
            raise ValueError("Written array contains trailing bytes")
        _verify_array_values(values, actual)

    def _timing_key(self, timing):
        previous = self._timings.get(timing.timeline_id)
        if previous is not None:
            key, existing = previous
            if not _same_timing(existing, timing):
                raise ValueError("Conflicting sample timing identity")
            return key
        key = f"timing_{len(self._timings):06d}"
        ids_file, times_file = f"{key}_sample_ids.npy", f"{key}_timestamps_s.npy"
        self._write_array(ids_file, timing.sample_ids)
        self._write_array(times_file, timing.timestamps_s)
        self._manifest["timings"].append(dict(timing_key=key, timeline_id=timing.timeline_id,
            clock_id=timing.clock_id, clock_epoch=timing.clock_epoch,
            source_identity=source_data(timing.source_identity),
            sample_ids_file=ids_file, timestamps_s_file=times_file))
        self._timings[timing.timeline_id] = (key, timing)
        return key

    def add_series(self, series):
        self._require_open()
        try:
            if not isinstance(series, TimeSeries):
                raise TypeError("add_series requires TimeSeries")
            series.__post_init__()
            return self._add_channel(series.data_tensor, series=series)
        except BaseException:
            self._abort()
            raise

    def add_static(self, tensor):
        self._require_open()
        try:
            return self._add_channel(tensor)
        except BaseException:
            self._abort()
            raise

    def _add_channel(self, tensor, *, series=None):
        validate_scientific_tensor(tensor)
        identity = ChannelIdentity.from_data(tensor if series is None else series)
        scientific_key = (identity.source_identity, identity.channel_kind, identity.reference_id)
        if scientific_key in self._channel_keys:
            raise ValueError("Duplicate scientific channel, including static/dynamic ambiguity")
        sample_axis = None if series is None else series.sample_axis_name
        axes = {axis.axis_name for axis in tensor.tensor_layout.axes}
        if series is None:
            if axes & self._sample_axes:
                raise ValueError("Static tensor must not declare a sample axis")
            self._static_axes.update(axes)
        else:
            if sample_axis in self._static_axes:
                raise ValueError("Static tensor conflicts with declared sample axis")
            self._sample_axes.add(sample_axis)
        timing_key = None if series is None else self._timing_key(series.sample_timing)
        prefix = f"channel_{len(self._manifest['channels']):06d}"
        filename = f"{prefix}_values.npy"
        self._write_array(filename, tensor.data_values)
        companions = []
        for index, companion in enumerate(tensor.companion_tensors):
            companion_file = f"{prefix}_companion_{index:06d}.npy"
            self._write_array(companion_file, companion.data_values)
            companions.append(dict(metadata_name=companion.metadata_name, values_file=companion_file,
                tensor_layout=layout_data(companion.tensor_layout, timing_key=timing_key, sample_axis_name=sample_axis)))
        entry = dict(storage_kind="static_tensor" if series is None else "time_series", values_file=filename,
            tensor_layout=layout_data(tensor.tensor_layout, timing_key=timing_key, sample_axis_name=sample_axis),
            scientific_provenance=asdict(tensor.scientific_provenance), companion_tensors=companions,
            sample_axis_name=sample_axis, timing_ref=timing_key)
        # asdict retains tuples, while the public manifest schema uses JSON lists.
        entry["scientific_provenance"]["input_refs"] = list(tensor.scientific_provenance.input_refs)
        self._manifest["channels"].append(entry)
        self._channel_keys.add(scientific_key)
        return deepcopy(entry)

    def _close_handles(self):
        try:
            if self._archive is not None:
                self._archive.close()
        finally:
            if self._archive_file is not None:
                self._archive_file.close()

    def _abort(self):
        self._state = "failed"
        try:
            self._close_handles()
        finally:
            if self.format == "npy":
                if self.path.is_symlink():
                    raise RuntimeError("Writer destination changed to a symbolic link")
                for target in reversed(self._created_files):
                    target.unlink(missing_ok=True)
                # Never delete recursively: another actor's files remain untouched.
                if self.path.is_dir() and not any(self.path.iterdir()):
                    self.path.rmdir()
            else:
                self.path.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._state == "failed":
            if exc_type is None:
                raise RuntimeError("Dataset writing was aborted")
            return False
        if exc_type is not None:
            self._abort()
            return False
        self._require_open()
        try:
            self._manifest["channels"].sort(key=lambda entry: entry["storage_kind"] == "static_tensor")
            expected = manifest_inventory(self._manifest)
            if expected != self._filenames | {"dataset.json"}:
                raise ValueError("Manifest inventory differs from written payloads")
            content = json.dumps(self._manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if self.format == "npy":
                target = self.path / "dataset.json"
                with target.open("xb") as stream:
                    self._created_files.append(target)
                    stream.write(content)
            else:
                self._archive.writestr("dataset.json", content)
            self._filenames.add("dataset.json")
            self._close_handles()
            self._state = "closed"
        except BaseException:
            self._abort()
            raise
        return False


def write_numpy_dataset(dataset, path, *, format="npy", compressed=True, verify=True):
    """Serialize an already selected scientific dataset without retaining copies."""
    if not isinstance(dataset, RecordingDataset):
        raise TypeError("Expected RecordingDataset")
    with NumpyDatasetWriter(path, recording_id=dataset.recording_id, run_id=dataset.run_id,
                            definition_documents=dataset.definition_documents,
                            format=format, compressed=compressed, verify=verify) as writer:
        for series in dataset.time_series:
            writer.add_series(series)
        for tensor in dataset.static_tensors:
            writer.add_static(tensor)
    return writer.manifest


class _DatasetReader:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = None

    def __enter__(self):
        if self.path.is_symlink():
            raise ValueError("Dataset path cannot be a symbolic link")
        if self.path.is_dir():
            files = tuple(self.path.iterdir())
            if any(item.is_symlink() or not item.is_file() for item in files):
                raise ValueError("Dataset directory must contain regular files only")
            self.inventory = {item.name for item in files}
        else:
            self.archive = ZipFile(self.path, "r")
            entries = self.archive.infolist()
            self.inventory = {item.filename for item in entries}
            if len(self.inventory) != len(entries) or any(
                item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG)
                for item in entries
            ):
                self.archive.close()
                raise ValueError("Duplicate or non-regular ZIP members")
        return self

    def __exit__(self, *args):
        if self.archive is not None:
            self.archive.close()

    def open(self, name):
        return (self.path / name).open("rb") if self.archive is None else self.archive.open(name)

    def array(self, name):
        with self.open(name) as stream:
            values = np.lib.format.read_array(stream, allow_pickle=False)
            if stream.read(1):
                raise ValueError("Unexpected trailing bytes after NPY array")
        return values


def read_numpy_dataset(path):
    """Validate and restore owned immutable scientific objects, then close all IO."""
    with _DatasetReader(path) as reader:
        if "dataset.json" not in reader.inventory:
            raise ValueError("Missing dataset.json")
        with reader.open("dataset.json") as stream:
            manifest = parse_json(stream.read())
        expected = manifest_inventory(manifest)
        inventory = reader.inventory - ({"README.md"} if reader.archive is None else set())
        if inventory != expected:
            raise ValueError("Dataset file inventory differs from manifest")
        timings, timeline_ids = {}, set()
        for item in manifest["timings"]:
            key = text_value(item["timing_key"])
            timeline_id = text_value(item["timeline_id"])
            if key in timings or timeline_id in timeline_ids:
                raise ValueError("Duplicate timing identity")
            sample_ids, timestamps = reader.array(item["sample_ids_file"]), reader.array(item["timestamps_s_file"])
            if (sample_ids.dtype.kind, sample_ids.dtype.itemsize) != ("i", 8) or (timestamps.dtype.kind, timestamps.dtype.itemsize) != ("f", 8):
                raise ValueError("Timing arrays require int64 IDs and float64 seconds")
            timings[key] = SampleTiming(timeline_id=timeline_id, clock_id=text_value(item["clock_id"]),
                clock_epoch=integer_value(item["clock_epoch"]), source_identity=source_from_data(item["source_identity"]),
                sample_ids=sample_ids, timestamps_s=timestamps)
            timeline_ids.add(timeline_id)
        series_items, static_items, used_timings = [], [], set()
        for item in manifest["channels"]:
            key, axis = item["timing_ref"], item["sample_axis_name"]
            is_series = item["storage_kind"] == "time_series"
            if is_series:
                text_value(key)
                text_value(axis)
                if key not in timings:
                    raise ValueError("Unknown channel timing reference")
                used_timings.add(key)
            elif key is not None or axis is not None:
                raise ValueError("Static tensors cannot have timing references")
            def decode_layout(value):
                return layout_from_data(value, timings=timings, timing_key=key, sample_axis_name=axis)
            companions = tuple(CompanionTensor(companion["metadata_name"], reader.array(companion["values_file"]),
                decode_layout(companion["tensor_layout"])) for companion in item["companion_tensors"])
            tensor = DataTensor(reader.array(item["values_file"]), decode_layout(item["tensor_layout"]),
                provenance_from_data(item["scientific_provenance"]), companions)
            validate_scientific_tensor(tensor)
            tensor = tensor.immutable_copy()
            if is_series:
                series_items.append(TimeSeries(tensor, timings[key], sample_axis_name=axis))
            else:
                static_items.append(tensor)
        if used_timings != set(timings):
            raise ValueError("Unreferenced timing registry entry")
        return RecordingDataset(manifest["recording_id"], manifest["run_id"], tuple(series_items), tuple(static_items),
                                definitions_from_data(manifest["definition_documents"]))
