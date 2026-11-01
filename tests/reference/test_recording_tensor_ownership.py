"""Exercise tensor ownership using an unchanged prepared recording as input."""

import json
from pathlib import Path

import numpy as np

from skellyforge.core.data_tensors import (
    AxisDefinition, ChannelIdentity, CompanionTensor, DataTensor, DefinitionDocument,
    RecordingDataset, SampleTiming, DataProvenance, SourceIdentity, TensorLayout,
    TimeSeries,
)
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def test_reference_dataset_preserves_values_timing_masks_and_shared_views(request, tmp_path):
    explicit = request.config.getoption("--parquet")
    dataset_name = request.config.getoption("--dataset")
    path = Path(explicit) if explicit else recording_path(dataset_name.removesuffix("_data"))
    original_hash = digest(path)
    records, scale_fit, metadata = read_recording(
        path, sensor_group=request.config.getoption("--sensor-group"), include_model=True,
    )
    channel = metadata["channel"]
    names = tuple(channel["names"])
    components = tuple(channel["components"])
    columns = {name: index for index, name in enumerate(names)}
    values = np.full((len(records), len(names), len(components)), np.nan)
    presence = np.zeros(values.shape, dtype=bool)
    # Existing Forge reader returns viewer records. Adapt those fixtures only;
    # this test does not define a new runtime reader or claim raw Parquet fidelity.
    for index, record in enumerate(records):
        if record["points"]:
            selected = [columns[name] for name in record["points"]]
            values[index, selected] = tuple(record["points"].values())
            presence[index, selected] = True
    source = SourceIdentity(source_id=channel["source"], sensor_group=channel["sensor_group"],
                            model_id="standard_human")
    sample_ids = np.asarray([record["number"] for record in records], dtype=np.int64)
    timestamps = np.asarray([record["time"] for record in records], dtype=np.float64)
    # Clock identity is supplied explicitly by this fixture, not inferred from
    # timestamp equality or advertised as recovered recording clock metadata.
    timing = SampleTiming(timeline_id="reference_fixture_samples", clock_id="reference_fixture_clock",
                          source_identity=source, sample_ids=sample_ids, timestamps_s=timestamps)
    axes = (timing.sample_axis, AxisDefinition("entity_identity", names),
            AxisDefinition("component_identity", components))
    layout = TensorLayout(axes, values.dtype, tuple(channel["components"].values()),
                          reference_id=channel["reference_frame"], source_identity=source,
                          channel_kind=channel["kind"])
    mask_layout = TensorLayout(axes, presence.dtype, source_identity=source)
    tensor = DataTensor(values, layout,
                        DataProvenance("saved_model_estimate", "reference_tensor_admission",
                                             (metadata["sha256"],), "standard_human"),
                        (CompanionTensor("entry_presence", presence, mask_layout),))
    assert tensor.data_values is values
    series = TimeSeries(tensor, timing).immutable_copy()
    segment_names = tuple(scale_fit.segment_lengths)
    lengths = np.asarray(tuple(scale_fit.segment_lengths.values()), dtype=np.float64)
    static_layout = TensorLayout(
        (AxisDefinition("entity_identity", segment_names), AxisDefinition("component_identity", ("length",))),
        lengths.dtype, ("mm",), reference_id=channel["reference_frame"], source_identity=source,
        channel_kind="SEGMENT_LENGTHS",
    )
    static = DataTensor(lengths[:, None], static_layout,
                        DataProvenance("fitted_lengths", "saved_scale_fit",
                                             (metadata["sha256"],), "standard_human")).immutable_copy()
    definition = DefinitionDocument("model:standard_human", json.dumps(metadata["model"], allow_nan=False))
    dataset = RecordingDataset(path.stem, metadata["run_id"], (series,), (static,), (definition,))
    assert dataset.time_series[0] is series
    assert dataset.static_tensors[0] is static
    assert dataset.sample_timings[0] is timing
    np.testing.assert_array_equal(series.data_tensor.data_values, values)
    np.testing.assert_array_equal(series.data_tensor.companion_values("entry_presence"), presence)
    np.testing.assert_array_equal(series.sample_timing.sample_ids, sample_ids)
    np.testing.assert_array_equal(series.sample_timing.timestamps_s, timestamps)
    assert series.data_tensor.tensor_layout.component_units == tuple(channel["components"].values())
    for start in range(0, series.sample_count, 37):
        selected = series.slice_samples(start, start + 37)
        np.testing.assert_array_equal(selected.data_tensor.data_values, values[start:start + 37])
        np.testing.assert_array_equal(selected.sample_timing.sample_ids, sample_ids[start:start + 37])
        np.testing.assert_array_equal(selected.sample_timing.timestamps_s, timestamps[start:start + 37])
        assert np.shares_memory(selected.data_tensor.data_values, series.data_tensor.data_values)
        assert np.shares_memory(selected.data_tensor.companion_values("entry_presence"),
                                series.data_tensor.companion_values("entry_presence"))
    selection = dataset.select_channels((ChannelIdentity.from_data(series),))
    assert selection.time_series[0] is series
    assert selection.definition_documents[0] is definition
    assert selection.static_tensors == ()
    for output_format in ("npy", "npz"):
        saved_path = tmp_path / output_format
        dataset.save_numpy(saved_path, format=output_format)
        restored = RecordingDataset.load_numpy(saved_path)
        assert restored.recording_id == dataset.recording_id
        assert restored.run_id == dataset.run_id
        assert restored.definition_documents == dataset.definition_documents
        restored_series = restored.time_series[0]
        assert restored_series.data_tensor.tensor_layout == layout
        assert restored_series.data_tensor.data_provenance == tensor.data_provenance
        np.testing.assert_array_equal(restored_series.data_tensor.data_values, values)
        np.testing.assert_array_equal(restored_series.data_tensor.companion_values("entry_presence"), presence)
        np.testing.assert_array_equal(restored_series.sample_timing.sample_ids, sample_ids)
        np.testing.assert_array_equal(restored_series.sample_timing.timestamps_s, timestamps)
        assert restored.static_tensors[0].tensor_layout == static_layout
        np.testing.assert_array_equal(restored.static_tensors[0].data_values, lengths[:, None])
        assert restored_series.is_immutable and restored.static_tensors[0].is_immutable
    assert digest(path) == original_hash == metadata["sha256"]
