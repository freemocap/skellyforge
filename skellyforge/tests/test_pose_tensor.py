"""Tensor admission, ownership and numerical equivalence across streaming/chunks."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor, SolutionCode, HydrationStatus
from skellyforge.core.skeleton.pose.hydration_batch import LandmarkPoseGeometry
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.pose.model_scale_fitting import StreamingModelScaleFitter, fit_model_scale
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, SegmentPose
from skellyforge.core.skeleton.linkage.joint_batch import JointBatchGeometry
from skellyforge.tests.pose_tensor_oracle import scalar_roll_oracle


def pose_sequence(sample_count=39):
    skeleton = SkeletonDefinition.from_default_yaml()
    # Exercise actual rigid-terminal backfill, beyond observation-frame branches.
    skeleton = replace(skeleton, segments=dict(skeleton.segments) | {
        "left_carpals": replace(skeleton.segments["left_carpals"], observation_frame=None)})
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    names = tuple(rest.landmark_positions)
    geometry = LandmarkPoseGeometry.from_skeleton(skeleton=skeleton, landmark_names=names)
    values = np.broadcast_to([rest.landmark_positions[name].array * 1700 for name in names],
                             (sample_count, len(names), 3)).copy()
    rng = np.random.default_rng(43211)
    values += rng.normal(0, 5, values.shape)
    present = rng.random(values.shape[:2]) > .08
    present[10:12] = False
    values[20] = 0.  # Degenerate observations remain distinct from missing ones.
    return skeleton, rest, geometry.recover_tensor(positions=values, landmark_present=present)


def test_numeric_hydration_ownership_and_safe_views():
    _, _, tensor = pose_sequence()
    assert all(values.dtype.kind != "O" for values in tensor.arrays())
    assert HydrationStatus.DEGENERATE in tensor.hydration_status
    assert np.all(tensor.solution_codes[~tensor.segment_present] == SolutionCode.MISSING)
    owned = tensor.copy_owned()
    assert owned.is_immutable and owned.copy_owned() is owned
    with patch('skellyforge.core.skeleton.pose.pose_tensor.np.isfinite', side_effect=AssertionError("Owned view rescanned values")):
        view = owned.sample_view(slice(2, 5))
    for original, selected in zip(owned.arrays(), view.arrays(), strict=True):
        assert np.shares_memory(original, selected)
        with pytest.raises(ValueError):
            selected.flags.writeable = True
    mutable = owned.copy_owned(writable=True)
    assert not mutable.is_immutable
    assert not np.shares_memory(mutable.origins, owned.origins)
    mutable.scale_estimates[0, mutable.segment_present[0]] = 0.
    with pytest.raises(ValueError, match="positive"):
        mutable.sample_view(slice(0, 1))


def test_pose_admission_rejects_incompatible_axes_values_and_metadata():
    _, _, tensor = pose_sequence()
    for field, value in (
        ('segment_names', tensor.segment_names[:-1]),
        ('segment_names', (tensor.segment_names[0],) * len(tensor.segment_names)),
        ('origins', tensor.origins.astype(np.float32)),
        ('orientations', tensor.orientations[..., :-1]),
        ('solution_codes', np.full(tensor.solution_codes.shape, 255, np.uint8)),
        ('solution_codes', tensor.solution_codes.astype(object)),
        ('hydration_status', np.full(tensor.hydration_status.shape, 255, np.uint8)),
    ):
        with pytest.raises(ValueError):
            replace(tensor, **{field: value})
    with pytest.raises(ValueError):
        tensor.sample_view(slice(None, None, -1))


def test_tensor_roll_matches_independent_scalar_and_chunked_calls_without_pose_objects():
    skeleton, rest, tensor = pose_sequence()
    resets = np.zeros(tensor.sample_count, bool)
    resets[11] = True
    expected_rotations, expected_codes = scalar_roll_oracle(skeleton=skeleton, rest=rest, poses=tensor, reset_before=resets)
    def resolver():
        return ContinuousRollResolver.for_skeleton(skeleton=skeleton, rest_relative_orientations=rest.relative_orientations)
    whole_resolver, chunk_resolver = resolver(), resolver()
    with patch.object(SkeletonPose, '__init__', side_effect=AssertionError("Pose object in numerical path")), \
         patch.object(SegmentPose, '__init__', side_effect=AssertionError("Segment object in numerical path")):
        whole = whole_resolver.resolve_tensor(poses=tensor, reset_before=resets)
        chunks = [chunk_resolver.resolve_tensor(poses=tensor.sample_view(slice(start, start + 7)),
                  reset_before=resets[start:start + 7]) for start in range(0, tensor.sample_count, 7)]
        joints = JointBatchGeometry.from_skeleton(skeleton=skeleton).calculate_tensor(poses=whole)
    np.testing.assert_allclose(whole.orientations, expected_rotations, atol=1e-11, rtol=0, equal_nan=True)
    np.testing.assert_array_equal(whole.solution_codes, expected_codes)
    np.testing.assert_array_equal(np.concatenate([part.orientations for part in chunks]), whole.orientations)
    assert joints.parent_codes.dtype == np.uint8 and joints.child_codes.dtype == np.uint8
    np.testing.assert_array_equal(tensor.orientations, pose_sequence()[2].orientations)
    with pytest.raises(ValueError, match="identities"):
        whole_resolver.resolve_tensor(poses=replace(tensor, segment_names=tuple(reversed(tensor.segment_names))))
    with pytest.raises(ValueError, match="Reset flags"):
        whole_resolver.resolve_tensor(poses=tensor, reset_before=np.zeros(tensor.sample_count, int))


def test_dense_scale_ring_matches_last_observed_values_and_ragged_adapter():
    skeleton, _, poses = pose_sequence(71)
    names, size = poses.segment_names, 5
    voter_names = frozenset(names[::2])
    frame_fitter = StreamingModelScaleFitter(skeleton=skeleton, voting_segment_names=voter_names, window_frames=size)
    chunk_fitter = StreamingModelScaleFitter(skeleton=skeleton, voting_segment_names=voter_names, window_frames=size)
    evidence = {name: [] for name in names}
    for sample in range(poses.sample_count):
        frame_fitter.observe_tensor(poses=poses.sample_view(slice(sample, sample + 1)))
        for column in np.flatnonzero(poses.segment_present[sample]):
            evidence[names[column]].append(poses.scale_estimates[sample, column])
            evidence[names[column]] = evidence[names[column]][-size:]
        if frame_fitter.has_model_scale:
            expected = fit_model_scale(skeleton=skeleton, scale_samples=evidence, voting_segment_names=voter_names)
            actual = frame_fitter.current_tensor()
            np.testing.assert_allclose(actual.segment_scales, [expected.segment_scales[name] for name in names], rtol=1e-14)
            assert actual.fitted_scale == expected.fitted_scale
    # One incoming chunk greatly exceeds the observation window.
    chunk_fitter.observe_tensor(poses=poses)
    np.testing.assert_array_equal(chunk_fitter.current_tensor().segment_scales, frame_fitter.current_tensor().segment_scales)
    chunk_fitter.reset()
    assert not chunk_fitter.has_model_scale


def test_joint_and_scale_results_reject_inconsistent_declarations_and_values():
    skeleton, _, tensor = pose_sequence()
    joints = JointBatchGeometry.from_skeleton(skeleton=skeleton).calculate_tensor(poses=tensor)
    for field, value in (
        ('joint_names', joints.joint_names[:-1]),
        ('angle_names', joints.angle_names[:-1]),
        ('angles', joints.angles[..., :-1]),
        ('angles', joints.angles.astype(np.float32)),
        ('parent_codes', np.full(joints.parent_codes.shape, 255, np.uint8)),
        ('child_codes', joints.child_codes.astype(object)),
        ('joint_present', joints.joint_present.astype(np.uint8)),
    ):
        with pytest.raises(ValueError):
            replace(joints, **{field: value})
    fitter = StreamingModelScaleFitter(skeleton=skeleton, voting_segment_names=frozenset(tensor.segment_names))
    fitter.observe_tensor(poses=tensor)
    fit = fitter.current_tensor()
    for field, value in (
        ('segment_names', tuple(reversed(fit.segment_names)) + ('extra',)),
        ('segment_scales', np.full_like(fit.segment_scales, np.inf)),
        ('segment_lengths', fit.segment_lengths[:-1]),
        ('measured_support', np.zeros_like(fit.measured_support)),
        ('voting_support', fit.voting_support.astype(np.uint8)),
    ):
        with pytest.raises(ValueError):
            replace(fit, **{field: value})


def test_mixed_object_and_tensor_calls_share_temporal_state():
    skeleton, rest, tensor = pose_sequence()
    def resolver():
        return ContinuousRollResolver.for_skeleton(skeleton=skeleton, rest_relative_orientations=rest.relative_orientations)
    expected, mixed = resolver(), resolver()
    for sample in range(tensor.sample_count):
        view = tensor.sample_view(slice(sample, sample + 1))
        # A missing sample with no reset is skipped by both pipeline callers.
        numerical = expected.resolve_tensor(poses=view)
        if sample % 2 and view.segment_present.any():
            actual = PoseTensor.from_pose(pose=mixed.resolve_pose(pose=view.to_pose()), segment_names=tensor.segment_names)
        else:
            actual = mixed.resolve_tensor(poses=view)
        np.testing.assert_allclose(actual.orientations, numerical.orientations, atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(actual.solution_codes, numerical.solution_codes)


def test_joint_tensor_accepts_empty_joint_models():
    skeleton, _, tensor = pose_sequence()
    skeleton = replace(skeleton, joints={}, chains={})
    result = JointBatchGeometry.from_skeleton(skeleton=skeleton).calculate_tensor(poses=tensor)
    assert result.joint_names == ()
    assert result.angles.shape == (tensor.sample_count, 0, 3)
    assert result.parent_codes.dtype == np.uint8
