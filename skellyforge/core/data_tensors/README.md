# Tensor ownership

SkellyForge owns these reusable numerical data types. They depend on NumPy and
the standard library; they do not import FreeMoCap or SkellyTracker.

- `DataTensor` composes a numerical array, `TensorLayout`, `DataProvenance`,
  and explicitly aligned `CompanionTensor` arrays. Components come from the
  declared layout, not a universal coordinate count.
- `SampleTiming` associates original sample IDs with timestamps, source identity,
  clock identity and clock epoch. It neither creates timing nor synchronizes clocks.
- `TimeSeries` composes a `DataTensor` and `SampleTiming`. Its `sample_timing`
  attribute replaces the former Core name `sample_timeline`.
- `RecordingDataset` collects immutable time series, static tensors and definition
  documents for one recording run. Its `sample_timings` property retains distinct
  clock domains. It does not require FreeMoCap's recording schema.
- `ChannelValues` exposes named access to a dense entity array; numerical values
  remain in arrays rather than dictionaries.

Admission borrows tensor arrays. `immutable_copy()` publishes independent storage;
subsequent contiguous selections share that immutable storage. `mutable_copy()`
explicitly allocates writable independent arrays. A frozen Python object alone
does not make a borrowed array immutable. These ownership rules are unchanged by
the package move.

FreeMoCap owns its pipeline composition, adapters from tracker/reconstruction
results, and publication into its Parquet recording schema. Those adapters must
supply actual timing, identities, units, references and provenance;
these classes do not infer missing context.

The [NumPy format contract](NUMPY_FORMAT.md) defines standalone NPY directories and
NPZ archives for selected datasets. `write_numpy_dataset`,
`read_numpy_dataset`, and `NumpyDatasetWriter` preserve the declared arrays and
metadata without importing producer packages. `RecordingDataset.save_numpy` and
`RecordingDataset.load_numpy` provide convenience access. Writers verify each
array before releasing borrowed channel storage; loaders return owned immutable
objects. Solver execution status and caches are outside this public format.

FreeMoCap imports these types through the workspace's existing editable dependency;
its duplicate implementations and generic contract tests have been removed. The
owner authorized local integration without commit/push pauses for this task.
Cross-package integration tests remain in FreeMoCap. Existing Forge pose, joint
and scale helpers retain their current interfaces; this stage does not rewrite
those algorithms or claim to have unified their internal storage.

Validation for this extraction: 75 tests and 28 subtests passed, combining the
generic tensor contracts, channel access, blocked producer imports, existing pose
contracts, and the prepared 222-frame reference recording. The input recording's
SHA-256 remained
`5dca07371cbd5410031f76283a758e099455b34c74498d47995897e6ffb04bd8`.
The reference test exercises the new containers through Forge's existing reader;
it does not certify complete Parquet-to-dataset metadata reconstruction. FreeMoCap
consumer validation is recorded in that repository's tensor-contract audit.
