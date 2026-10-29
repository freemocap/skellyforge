"""Exercise numerical pose interfaces on saved reference data, without producers."""
from pathlib import Path

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.skeleton.linkage.joint_batch import JointBatchGeometry
from skellyforge.core.skeleton.pose.hydration_batch import LandmarkPoseGeometry
from skellyforge.core.skeleton.pose.model_scale_fitting import StreamingModelScaleFitter
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.skeleton_snapshot import RestPoseSnapshot, SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def recording_inputs(request):
    dataset = request.config.getoption('--dataset')
    explicit = request.config.getoption('--parquet')
    path = Path(explicit) if explicit else recording_path(dataset.removesuffix('_data'))
    records, scale, metadata = read_recording(path,
        sensor_group=request.config.getoption('--sensor-group'), include_model=True,
        landmark_kind='MAPPED_KEYPOINTS_3D')
    skeleton = SkeletonSnapshot.from_dict(metadata['skeleton']).restore()
    definition = metadata['model']['rest_pose']
    rest = RestPoseSnapshot(**{**definition, 'orientations': {
        name: RotationQuaternion.from_components(**value)
        for name, value in definition['orientations'].items()
    }}).restore(skeleton=skeleton)
    names = tuple(skeleton.landmarks)
    indices = {name: index for index, name in enumerate(names)}
    positions = np.full((len(records), len(names), len(('x', 'y', 'z'))), np.nan)
    for index, record in enumerate(records):
        if record['points']:
            positions[index, [indices[name] for name in record['points']]] = tuple(record['points'].values())
    assert len(records) == {'test_data': 222, 'sample_data': 1108}[dataset]
    times = np.asarray([record['time'] for record in records])
    assert np.isfinite(times).all() and (np.diff(times) > 0).all()
    geometry = LandmarkPoseGeometry.from_skeleton(skeleton=skeleton, landmark_names=names)
    return path, records, scale, metadata, skeleton, rest, geometry, positions


def test_reference_tensor_roll_matches_saved_orientations_and_chunked_execution(request):
    path, records, _, metadata, skeleton, rest, geometry, positions = recording_inputs(request)
    present = np.isfinite(positions).all(axis=-1)
    recovered = geometry.recover_tensor(positions=positions, landmark_present=present)
    reset_before = ~present.any(axis=-1)

    def resolver():
        return ContinuousRollResolver.for_skeleton(skeleton=skeleton,
            rest_relative_orientations=rest.relative_orientations)

    resolved = resolver().resolve_tensor(poses=recovered, reset_before=reset_before)
    indices = {name: index for index, name in enumerate(resolved.segment_names)}
    compared = 0
    for index, record in enumerate(records):
        if not record['rotations']:
            continue
        columns = [indices[name] for name in record['rotations']]
        assert resolved.segment_present[index, columns].all()
        expected = np.asarray(tuple(record['rotations'].values()))
        # Quaternion sign is representational; compare physical orientation.
        dots = np.abs(np.sum(expected * resolved.orientations[index, columns], axis=-1))
        np.testing.assert_allclose(dots, 1., atol=1e-10, rtol=0)
        compared += len(columns)
    assert compared > 0
    joint_geometry = JointBatchGeometry.from_skeleton(skeleton=skeleton)
    joints = joint_geometry.calculate_tensor(poses=resolved)
    for size in (1, 37):
        chunk_resolver = resolver()
        for start in range(0, len(records), size):
            stop = min(start + size, len(records))
            chunk = chunk_resolver.resolve_tensor(poses=recovered.sample_view(slice(start, stop)),
                reset_before=reset_before[start:stop])
            np.testing.assert_allclose(chunk.orientations, resolved.orientations[start:stop], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(chunk.solution_codes, resolved.solution_codes[start:stop])
            chunk_joints = joint_geometry.calculate_tensor(poses=chunk)
            np.testing.assert_allclose(chunk_joints.angles, joints.angles[start:stop], atol=1e-12, rtol=0)
    assert digest(path) == metadata['sha256']


def test_reference_tensor_scale_evidence_matches_frame_and_batch_ingestion(request):
    path, records, scale, metadata, skeleton, _, geometry, positions = recording_inputs(request)
    poses = geometry.recover_tensor(positions=positions, landmark_present=np.isfinite(positions).all(axis=-1))

    def fitter():
        return StreamingModelScaleFitter(skeleton=skeleton,
            voting_segment_names=scale.voting_segment_names, window_frames=17)

    frame_fitter, batch_fitter = fitter(), fitter()
    for start in range(0, len(records), 37):
        stop = min(start + 37, len(records))
        batch_fitter.observe_tensor(poses=poses.sample_view(slice(start, stop)))
        for index in range(start, stop):
            frame_fitter.observe_tensor(poses=poses.sample_view(slice(index, index + 1)))
        assert frame_fitter.has_model_scale == batch_fitter.has_model_scale
        if frame_fitter.has_model_scale:
            frame_fit, batch_fit = frame_fitter.current_tensor(), batch_fitter.current_tensor()
            assert frame_fit.segment_names == batch_fit.segment_names
            np.testing.assert_allclose(frame_fit.fitted_scale, batch_fit.fitted_scale, atol=1e-12, rtol=0)
            np.testing.assert_allclose(frame_fit.segment_scales,
                batch_fit.segment_scales, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(frame_fit.measured_support, batch_fit.measured_support)
    assert digest(path) == metadata['sha256']
