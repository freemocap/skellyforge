"""The reusable tensor API operates when producer packages cannot be imported."""

import subprocess
import sys


def test_tensor_package_does_not_import_sibling_apps(tmp_path):
    program = '''
import importlib.abc
import sys
from pathlib import Path

class RejectProducerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'freemocap', 'skellytracker', 'skellycam'}:
            raise AssertionError(f'Forbidden producer dependency: {fullname}')

sys.meta_path.insert(0, RejectProducerImports())
import numpy as np
from skellyforge.core.data_tensors import (
    AxisDefinition, ChannelValues, DataTensor, RecordingDataset, SampleTiming,
    ScientificProvenance, SourceIdentity, TensorLayout, TimeSeries,
)
identity = SourceIdentity('independent_sensor')
timing = SampleTiming(timeline_id='capture', clock_id='sensor_clock',
    source_identity=identity, sample_ids=[5, 9], timestamps_s=[0.1, 0.7])
layout = TensorLayout((timing.sample_axis,), 'float64', source_identity=identity,
    channel_kind='independent_signal')
series = TimeSeries(DataTensor(np.array([2., 4.]), layout,
    ScientificProvenance('model_prediction', 'standalone_test')).immutable_copy(), timing)
dataset = RecordingDataset('independent_recording', 0, (series,))
assert dataset.time_series[0].frame_view(1).sample_timing.sample_ids[0] == 9
for output_format in ('npy', 'npz'):
    path = Path(sys.argv[1]) / output_format
    dataset.save_numpy(path, format=output_format)
    restored = RecordingDataset.load_numpy(path)
    np.testing.assert_array_equal(restored.time_series[0].data_tensor.data_values, [2., 4.])
assert not any(name.split('.')[0] in {'freemocap', 'skellytracker', 'skellycam'} for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-B", "-c", program, str(tmp_path)],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
