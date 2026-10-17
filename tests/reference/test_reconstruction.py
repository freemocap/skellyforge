"""Rehydrate actual mapped observations; never run a connected optimizer."""
from pathlib import Path

import numpy as np

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def test_recording_hydration(request):
    dataset = request.config.getoption('--dataset')
    explicit = request.config.getoption('--parquet')
    path = Path(explicit) if explicit else recording_path(dataset.removesuffix('_data'))
    before = digest(path)
    records, scale, metadata = read_recording(path,
        sensor_group=request.config.getoption('--sensor-group'),
        landmark_kind='MAPPED_KEYPOINTS_3D')
    expected = {'test_data': 222, 'sample_data': 1108}[dataset]
    assert len(records) == expected
    times = np.asarray([record['time'] for record in records])
    assert np.isfinite(times).all() and (np.diff(times) > 0).all()
    skeleton = SkeletonSnapshot.from_dict(metadata['skeleton']).restore()
    assert set(scale.segment_scales) == set(skeleton.segments)
    assert all(np.isfinite(value) and value > 0 for value in scale.segment_scales.values())
    populated = 0
    empty = 0
    for record in records:
        observed = {name: Point(np.asarray(value)) for name, value in record['points'].items()}
        pose = hydrate_skeleton(skeleton=skeleton, observed=observed, require_all=False)
        if not observed:
            assert not pose.segment_poses
            empty += 1
        if pose.segment_poses:
            populated += 1
        for segment in pose.segment_poses.values():
            assert np.isfinite(segment.origin.array).all()
            assert np.isfinite(segment.scale_estimate) and segment.scale_estimate > 0
            np.testing.assert_allclose(np.linalg.norm(segment.orientation.as_array()), 1., atol=1e-6)
    assert populated > 0, 'No frame produced a reconstructed segment'
    assert digest(path) == before, 'Reference recording changed during reconstruction'
    print(f'{dataset}: {len(records)} frames; hydrated={populated}, absent={empty}; SHA256 {before}')
