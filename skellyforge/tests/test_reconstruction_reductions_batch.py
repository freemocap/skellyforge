"""Batch reductions retain scalar missingness and accumulation order."""
import numpy as np

from skellyforge.core.biomechanics.center_of_mass import CenterOfMassDefinitions, compute_segment_coms
from skellyforge.core.biomechanics.center_of_mass_batch import CenterOfMassGeometry
from skellyforge.core.biomechanics.composite_inertia import whole_body_center_of_mass
from skellyforge.core.skeleton.pose.rigid_body_diagnostics import ResidualBatchGeometry, measure_rigid_body_residuals
from skellyforge.core.skeleton.pose.hydration_batch import LandmarkPoseGeometry
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.math.kinematics.rigid_point_set import align_point_sets_batch, align_point_sets_similarity
from skellyforge.core.skeleton.pose.model_scale_fitting import fit_model_scale_arrays, StreamingModelScaleFitter
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, SegmentPose, PoseSolution
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion


def test_com_and_residuals_match_scalar():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    names = tuple(rest.landmark_positions)
    positions = np.broadcast_to([rest.landmark_positions[n].array * 1700 for n in names], (15, len(names), 3)).copy()
    present = np.random.default_rng(347).random(positions.shape[:2]) > .3
    present[0] = False
    definitions = CenterOfMassDefinitions.from_default_yaml()
    masses = {n: .25 + i for i, n in enumerate(n for d in definitions.definitions.values() for n, _ in d.side_entries)}
    com_geometry = CenterOfMassGeometry.from_definitions(definitions=definitions, landmark_names=names, segment_masses=masses)
    com = com_geometry.calculate(positions=positions, landmark_present=present)
    recovery = LandmarkPoseGeometry.from_skeleton(skeleton=skeleton, landmark_names=names)
    poses = recovery.recover(positions=positions, landmark_present=present)
    measured = frozenset(recovery.segment_names[::2])
    references = {n: skeleton.segments[n].length * 1650 for n in recovery.segment_names[::3]}
    residual_geometry = ResidualBatchGeometry.from_skeleton(skeleton=skeleton, measured_segment_names=measured, reference_lengths=references, reference_kind="test")
    lengths, residuals = residual_geometry.calculate(scales=poses.scale_estimates, segment_present=poses.segment_present)
    for frame in range(len(positions)):
        world = {n: positions[frame, i] for i, n in enumerate(names) if present[frame, i]}
        expected_coms = compute_segment_coms(definitions=definitions, world=world)
        expected_body = whole_body_center_of_mass(segment_coms=expected_coms, segment_masses=masses)
        assert com.body_present[frame] == (expected_body is not None)
        if expected_body is not None:
            np.testing.assert_array_equal(com.body_positions[frame], expected_body)
        for column, name in enumerate(com_geometry.segment_names):
            assert com.segment_present[frame, column] == (name in expected_coms)
            if name in expected_coms:
                np.testing.assert_array_equal(com.segment_positions[frame, column], expected_coms[name])
        pose = hydrate_skeleton(skeleton=skeleton, observed={n: Point.from_array(values=p) for n, p in world.items()}, require_all=False)
        expected = measure_rigid_body_residuals(skeleton=skeleton, pose=pose, measured_segment_names=measured,
            reference_lengths=references, reference_kind="test")
        for column, name in enumerate(residual_geometry.segment_names):
            for actual, value in ((lengths[frame, column], expected[name].measured_length), (residuals[frame, column], expected[name].residual)):
                if value is None:
                    assert np.isnan(actual)
                else:
                    np.testing.assert_array_equal(actual, value)


def test_rigid_batch_matches_scalar_and_marks_degenerate_fit():
    rng = np.random.default_rng(765)
    reference = rng.normal(size=(9, 3))
    observed = rng.normal(size=(23, 9, 3)) * 1000
    observed[0] = 0
    observed[1] = reference * [-1, 1, 1]  # Reflection correction must match.
    matrices, translations, scales, valid = align_point_sets_batch(reference=reference, observed=observed)
    assert not valid[0]
    assert np.isnan(scales[0])
    for frame in range(1, len(observed)):
        expected = align_point_sets_similarity(reference=Point(reference), observed=Point(observed[frame]))
        np.testing.assert_allclose(matrices[frame], expected.transform.rotation.to_rotation_matrix(), atol=1e-14, rtol=0)
        np.testing.assert_allclose(translations[frame], expected.transform.translation.array, atol=1e-10, rtol=0)
        np.testing.assert_allclose(scales[frame], expected.scale, atol=1e-10, rtol=0)


def test_recording_fit_arrays_preserve_first_observation_order_and_evidence():
    skeleton = SkeletonDefinition.from_default_yaml()
    names = tuple(skeleton.segments)
    rng = np.random.default_rng(94)
    scales = rng.uniform(1500, 1800, (19, len(names)))
    present = rng.random(scales.shape) > .35
    present[:3, ::2] = False
    voters = frozenset(names)
    fitter = StreamingModelScaleFitter(skeleton=skeleton, voting_segment_names=voters, window_frames=len(scales))
    for row in range(len(scales)):
        fitter.observe_pose(pose=SkeletonPose(segment_poses={n: SegmentPose(segment_name=n,
            origin=Point.from_xyz(x=0., y=0., z=0.), orientation=RotationQuaternion.identity(),
            scale_estimate=float(scales[row, i]), solved_by=PoseSolution.DIRECTION)
            for i, n in enumerate(names) if present[row, i]}))
    result = fit_model_scale_arrays(skeleton=skeleton, segment_names=names, scales=scales, segment_present=present, voting_segment_names=voters)
    expected = fitter.current_fit()
    assert result.fitted_scale == expected.fitted_scale
    assert result.segment_scales == expected.segment_scales
    assert result.segment_lengths == expected.segment_lengths
    assert result.measured_segment_names == expected.measured_segment_names
