import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from skellyforge.core.data_tensors import RecordingDataset
from skellyforge.tests.test_numpy_dataset import example_dataset, assert_tensor_equal


@pytest.mark.parametrize('format', ['npy', 'npz'])
def test_original_v1_dataset_loads_and_resaves_as_v2(tmp_path, format):
    fixture = Path(__file__).parent / 'fixtures/numpy_dataset/provenance_v1.npz'
    old = fixture
    if format == 'npy':
        old = tmp_path / 'old'
        with ZipFile(fixture) as archive:
            archive.extractall(old)
    restored = RecordingDataset.load_numpy(old)
    expected = example_dataset()
    assert restored.channel_identities == expected.channel_identities
    for first, second in zip(restored.time_series, expected.time_series, strict=True):
        assert_tensor_equal(first.data_tensor, second.data_tensor)
        assert first.sample_timing.timeline_id == second.sample_timing.timeline_id
    for first, second in zip(restored.static_tensors, expected.static_tensors, strict=True):
        assert_tensor_equal(first, second)
    target = tmp_path / ('new' if format == 'npy' else 'new.npz')
    manifest = restored.save_numpy(target, format=format)
    assert manifest['version'] == 2
    assert all('data_provenance' in channel and 'scientific_provenance' not in channel
               for channel in manifest['channels'])
    again = RecordingDataset.load_numpy(target)
    assert again.channel_identities == restored.channel_identities


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('mutation', ['wrong_key', 'both_keys'])
def test_provenance_schema_cannot_mix_versions(tmp_path, version, mutation):
    root = tmp_path / 'dataset'
    manifest = example_dataset().save_numpy(root, format='npy')
    manifest['version'] = version
    if version == 1:
        for channel in manifest['channels']:
            channel['scientific_provenance'] = channel.pop('data_provenance')
    channel = manifest['channels'][0]
    right, wrong = ('scientific_provenance', 'data_provenance') if version == 1 else ('data_provenance', 'scientific_provenance')
    channel[wrong] = channel[right]
    if mutation == 'wrong_key':
        del channel[right]
    (root / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='exact manifest fields'):
        RecordingDataset.load_numpy(root)
