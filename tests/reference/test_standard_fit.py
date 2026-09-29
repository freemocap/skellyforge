"""One complete recording through the public accepted fit, without variant sweeps."""
from pathlib import Path

import numpy as np

from skellyforge.core.skeleton.fitting import fit_human
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def test_standard_fit(request):
    dataset = request.config.getoption('--dataset')
    explicit = request.config.getoption('--parquet')
    path = Path(explicit) if explicit else recording_path(dataset.removesuffix('_data'))
    before = digest(path)
    records, scale, metadata = read_recording(
        path, sensor_group=request.config.getoption('--sensor-group'), include_model=True)
    expected = {'test_data': 222, 'sample_data': 1108}[dataset]
    assert len(records) == expected, f'{dataset}: expected {expected} frames in {path}'
    times = np.asarray([record['time'] for record in records])
    assert np.isfinite(times).all() and (np.diff(times) > 0).all()
    skeleton = SkeletonSnapshot.from_dict(metadata['skeleton']).restore()
    print(f'Fitting {dataset}: {path} ({len(records)} frames, SHA256 {before})', flush=True)
    fitted = fit_human(skeleton, metadata['model'], scale.segment_scales, records)
    segments = len(fitted.model['names'])
    for name, width in (('translations', 3), ('quaternions', 4), ('linkage_displacements', 3)):
        values = np.asarray(getattr(fitted.sequence, name))
        assert values.shape == (expected, segments, width), name
        assert np.isfinite(values).all(), name
    np.testing.assert_allclose(np.linalg.norm(fitted.sequence.quaternions, axis=-1), 1., atol=1e-6)
    lengths = np.asarray(fitted.sequence.lengths)
    assert lengths.shape == (expected, segments)
    assert np.isfinite(lengths).all()
    assert digest(path) == before, 'Reference recording changed during the fit'
