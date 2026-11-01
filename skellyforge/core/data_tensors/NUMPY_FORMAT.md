# NumPy dataset format

Version 1 serializes the selected contents of `RecordingDataset`.
It preserves values, declared layouts, identities, timing, quality/evidence,
units, references, provenance and definition documents. It does not dump
arbitrary Python objects. Parquet remains FreeMoCap's durable recording; this
format neither changes Parquet nor requires the two representations to coincide.

FreeMoCap's adapter retains authored components (including `visibility`) and
maps a selected `SEGMENT_POSES` convenience output to its existing
`SEGMENT_ORIGINS` and `ROTATIONS_WORLD` runtime channels. Wide CSV may keep the
combined convenience projection. Selected model outputs include the necessary
static fits and definitions. Those choices belong in Core, not this serializer.

Core stages this format through its existing selected-run publication: a stable
`exports/npy/` directory for the normal run, with other explicitly selected runs
under their existing `run-<id>/` export directory. Same-run replacement retires
only obsolete managed NPY files after verifying their previous hashes. Standalone
Forge writers still require fresh destinations. Core's outer inventory describes
NPY and NPZ separately under `numpy_datasets`, so an unselected format retains its
own selection and source revision. Core's optional `runtime_dataset` argument
accepts an already selected dataset for NPY/NPZ; its saved-result path remains a
valid Parquet-to-runtime adapter. This generic save/load API needs no Parquet file.

## Storage layout

An NPY export is a new directory containing `dataset.json` and flat `.npy`
files. An NPZ export is a new ZIP file with the same member names and contents:
`dataset.json` is plain UTF-8 JSON, not an object array. Array entries are standard
NPY files written and read with `allow_pickle=False`. NPZ uses DEFLATE by default;
`compressed=False` selects stored ZIP entries. Compression does not change meaning.

Generated filenames never incorporate source/entity identifiers. Examples are
`timing_000000_sample_ids.npy`, `timing_000000_timestamps_s.npy`,
`channel_000000_values.npy`, and `channel_000000_companion_000000.npy`.
Every filename is a unique basename, without directories, drives, separators,
parent traversal or symbolic links. `dataset.json` and exactly its referenced
array files constitute the payload inventory; duplicate ZIP members, missing
files, unreferenced files and reused payload filenames are errors. NPY directories
may additionally contain exactly one regular `README.md` documentation file;
its contents are not part of payload validation. No other extra files or links
are allowed. NPZ retains the exact payload inventory. Shared timing uses a
registry reference, not multiple payload-file references.

## Manifest schema

All listed fields are required; nullable fields are written as JSON `null`.
Unknown fields and format versions are rejected. Arrays below are JSON lists
unless explicitly identified as NPY files. JSON objects reject duplicate keys
and non-finite JSON numbers. Definition text remains a JSON string containing
its original valid JSON content; it is not parsed and reformatted on save.

| Object | Fields |
| --- | --- |
| Root | `format: "skellyforge.numpy_dataset"`, `version: 2`, `recording_id: string`, `run_id: nonnegative integer`, `definition_documents: Definition[]`, `timings: Timing[]`, `channels: Channel[]` |
| Definition | `definition_id: string`, `content_json: string` |
| Source | `source_id: string`, `instance_id: nonnegative integer`, `sensor_group: string or null`, `camera_id: string or null`, `model_id: string or null` |
| Timing | `timing_key: string`, `timeline_id: string`, `clock_id: string`, `clock_epoch: nonnegative integer`, `source_identity: Source`, `sample_ids_file: basename`, `timestamps_s_file: basename` |
| Layout | `axes: Axis[]`, `dtype: string`, `component_units: string[]`, `reference_id: string or null`, `source_identity: Source or null`, `channel_kind: string or null` |
| Axis | Exactly `axis_name: string` plus either `identifiers: string[] or integer[]` or `timing_ref: string` |
| Provenance | `meaning: string`, `operation_id: string`, `input_refs: string[]`, `model_id: string or null` |
| Companion | `metadata_name: string`, `values_file: basename`, `tensor_layout: Layout` |
| Channel | `storage_kind: "time_series" or "static_tensor"`, `values_file: basename`, `tensor_layout: Layout`, `data_provenance: Provenance`, `companion_tensors: Companion[]`, `sample_axis_name: string or null`, `timing_ref: string or null` |

`timing_key` is an archive-local identifier such as `timing_000000`; it is distinct
from the retained `timeline_id`. Channels retain time-series order
followed by static-tensor order. Definition and companion order are preserved.
Channel identity is derived from source identity, channel kind, reference and
static/dynamic status using `RecordingDataset` rules, not array filenames.

`dtype` is NumPy's scalar `dtype.str`, including byte order; only the existing
`TensorLayout` boolean, integer, floating and complex types are permitted.
Object, structured and subarray dtypes are rejected. Shape is derived from the
ordered axis identifier counts and checked against the NPY header. No coordinate
width, rank, component order or units are inferred from a channel name.

For a time series, `timing_ref` selects its timing record and `sample_axis_name`
selects exactly one layout axis. That axis uses the same `timing_ref` instead of
duplicating sample IDs in JSON. Companion axes that include that sample axis use
the same reference. Other axes contain explicit identifiers. Static channels
set both timing fields to `null`, contain no timing-reference axes, and receive
no artificial frame dimension. Empty axes and static scalar tensors are valid.

Timing files contain one-dimensional `int64` sample IDs and `float64` timestamps
in seconds, with the existing `SampleTiming` validation. Either byte order is
accepted for these fixed-width timing files; `SampleTiming` restores native byte
order while preserving values and signed zeros. Tensor and companion arrays retain
their exact declared byte order. Shared timing is reused
only for matching declared `timeline_id` and exact source, clock, epoch, IDs and
times. Conflicts fail. Different timeline identities remain separate even when
their numerical arrays are equal. There is no resampling, clock synchronization
or requirement that channels have equal frame counts.

## Selection and round trips

Version 1 allows `entry_presence`, `value_validity`, `confidence_scores`,
`measured_support` and `voting_support` companions. All except confidence are
boolean; confidence is floating point. Parent/companion axis associations and
validity constraints follow `DataTensor`; presence and validity use the complete
parent layout, while confidence/evidence may use matching subset axes.
Unknown companions, including `solver_status` or hydration execution state,
raise an error. Callers select content before writing; the writer
never silently discards fields. Storage-row presence is not inferred to mean
validity, and unavailable evidence is not invented on reload.

Round trips preserve dtype, shape, logical values, NaN positions, signed zero,
all declared identities/context and explicit companions. They do not promise
Python object identity, strides, aliasing, caches or NaN payload bits. Definition
text is preserved exactly. Absent optional companions remain absent. Loading
returns owned immutable tensors and timing arrays, with shared timing objects
for registry references and no open reader or memory-map lifetime obligation.

## Python interface and bounded writing

The implementation target is:

```python
def write_numpy_dataset(dataset, path, *, format="npy", compressed=True, verify=True): ...
def read_numpy_dataset(path): ...  # Returns RecordingDataset.

with NumpyDatasetWriter(path, recording_id=recording_id, run_id=run_id,
                        definition_documents=definitions,
                        format="npz", compressed=True, verify=True) as writer:
    writer.add_series(time_series)
    writer.add_static(static_tensor)
```

The convenience writer traverses a `RecordingDataset`; the context writer admits
one complete channel at a time, including borrowed arrays. The caller keeps each
array valid and unchanged until its add call returns. The writer validates and
writes that channel immediately, retains only manifest/timing information, and
allows its numerical buffers to be released before the next channel. This is
channel streaming, not appending frames to an existing channel. Manifest ordering
groups series then static channels while preserving insertion order within each.

With `verify=True` (the default), each array is read back immediately and checked
against its still-borrowed source for dtype, shape, numerical values, NaN positions
and signed zeros. Verification retains only one additional array at a time.
`RecordingDataset.save_numpy(...)` and `RecordingDataset.load_numpy(path)` provide
the same writer and owned-object reader as convenience methods.

Numerical writes use NumPy/ZIP streaming operations, without Python loops over
samples, entities or scalar values. Metadata/channel/companion loops are allowed.
Readers validate manifest structure, file inventory, NPY dtype/shape, source and
channel uniqueness, timing relationships and the reconstructed object contracts.

Destinations must not exist. Write the manifest only after successful channel
writes; failure removes only files created in the writer-owned destination.
Existing user files are never replaced or recursively removed. FreeMoCap's
existing staged publication controls replacement of previously published exports.

## Plain NumPy access

No FreeMoCap or SkellyForge import is required to read the documented data:

```python
import json
from pathlib import Path
import numpy as np

directory = Path("recording_numpy")
manifest = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
channel = next(item for item in manifest["channels"] if item["storage_kind"] == "time_series")
timing = next(item for item in manifest["timings"] if item["timing_key"] == channel["timing_ref"])
values = np.load(directory / channel["values_file"], allow_pickle=False, mmap_mode="r")
sample_ids = np.load(directory / timing["sample_ids_file"], allow_pickle=False)
timestamps_s = np.load(directory / timing["timestamps_s_file"], allow_pickle=False)
```

For NPZ, read the same manifest and named NPY member without extracting files:

```python
from zipfile import ZipFile

with ZipFile("recording_numpy.npz") as archive:
    manifest = json.loads(archive.read("dataset.json"))
    channel = manifest["channels"][0]
    with archive.open(channel["values_file"]) as member:
        values = np.load(member, allow_pickle=False)
```

These examples assume a trusted export; `read_numpy_dataset` performs the full
contract validation described above. Plain NPY consumers may explicitly choose
memory mapping, independently of the owned-object loader.

## Provenance migration

Writers emit version 2 with `data_provenance`. Readers also accept version 1,
whose channel field is `scientific_provenance`, and construct `DataProvenance`
objects. Saving a loaded version-1 dataset writes version 2. The source dataset
is never rewritten during loading. Each version requires its exact field set;
mixed or duplicate provenance keys are rejected. Array values, identities,
timelines, units, and companion tensors keep the same representation.
