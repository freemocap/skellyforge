"""Joint shoulder fitting: fixed geometry, explicit preferences, honest residuals."""

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    fit_connected_pose,
    LandmarkTarget,
    RootPoseTolerances,
)
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def setup():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    names = frozenset(
        ("pelvis", "sacrolumbar", "thoracic", "left_clavicle", "right_clavicle")
    )
    fit = ModelScaleFit(
        fitted_scale=1700.0,
        segment_scales=dict.fromkeys(skeleton.segments, 1700.0),
        segment_lengths={n: s.length * 1700 for n, s in skeleton.segments.items()},
        measured_segment_names=frozenset(),
        voting_segment_names=frozenset(),
    )
    args = dict(
        skeleton=skeleton,
        fit=fit,
        segment_names=names,
        segment_relative_orientations={
            n: rest.relative_orientations[n] for n in names if n != "pelvis"
        },
        root_origin=Point.from_xyz(x=100.0, y=-200.0, z=800.0),
        root_world_orientation=RotationQuaternion.identity(),
    )
    return args


def test_free_root_recovers_rigid_motion_without_changing_connections():
    args = setup()
    moved = {
        **args,
        "root_origin": Point.from_xyz(x=240.0, y=-160.0, z=920.0),
        "root_world_orientation": RotationQuaternion.from_rotation_vector(
            rotation_vector=np.array([0.15, -0.2, 0.3])
        ),
    }
    _, _, desired = synthesize_fitted_pose(**moved)
    targets = {
        n: LandmarkTarget(position=desired[n], tolerance=0.1)
        for n in (
            "left_hip_socket",
            "right_hip_socket",
            "left_acromion",
            "right_acromion",
        )
    }
    result = fit_connected_pose(
        **args,
        targets=targets,
        rotation_tolerances_radians={},
        root_tolerances=RootPoseTolerances(1000.0, 1.0),
    )
    assert max(result.target_errors.values()) < 0.001
    assert_geometry(
        {
            **args,
            "root_origin": result.world_origins["pelvis"],
            "root_world_orientation": result.world_orientations["pelvis"],
        },
        result,
    )
    assert (
        result.world_orientations["pelvis"].angle_to(
            other=moved["root_world_orientation"]
        )
        < 1e-5
    )


@pytest.mark.parametrize(
    "translation,rotation", [(0.0, 1.0), (1.0, -1.0), (np.inf, 1.0), (1.0, np.nan)]
)
def test_root_priors_require_valid_scales(translation, rotation):
    with pytest.raises(ValueError, match="Root tolerances"):
        RootPoseTolerances(translation, rotation)


def test_free_root_solution_is_equivariant_to_world_transform_and_units():
    args = setup()
    _, _, points = synthesize_fitted_pose(**args)
    desired = {
        n: Point.from_array(values=points[n].array + np.array([40.0, -30.0, 70.0]))
        for n in (
            "left_hip_socket",
            "right_hip_socket",
            "left_acromion",
            "right_acromion",
        )
    }

    def run(arguments, observations, unit):
        return fit_connected_pose(
            **arguments,
            targets={n: LandmarkTarget(p, unit) for n, p in observations.items()},
            rotation_tolerances_radians=dict.fromkeys(
                arguments["segment_relative_orientations"], 0.5
            ),
            root_tolerances=RootPoseTolerances(1000 * unit, 1.0),
        )

    first = run(args, desired, 1.0)
    q = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.3, -0.4, 0.2])
    )
    unit = 0.001
    t = np.array([0.2, -0.1, 0.8])
    transform = lambda p: Point.from_array(
        values=q.rotate_vector(vector=p.array) * unit + t
    )
    original = args["fit"]
    scaled = ModelScaleFit(
        fitted_scale=original.fitted_scale * unit,
        segment_scales={n: v * unit for n, v in original.segment_scales.items()},
        segment_lengths={n: v * unit for n, v in original.segment_lengths.items()},
        measured_segment_names=original.measured_segment_names,
        voting_segment_names=original.voting_segment_names,
    )
    second = run(
        {
            **args,
            "fit": scaled,
            "root_origin": transform(args["root_origin"]),
            "root_world_orientation": q * args["root_world_orientation"],
        },
        {n: transform(p) for n, p in desired.items()},
        unit,
    )
    for n in args["segment_names"]:
        np.testing.assert_allclose(
            second.world_origins[n].array,
            transform(first.world_origins[n]).array,
            atol=1e-6,
        )
        assert (
            second.world_orientations[n].angle_to(other=q * first.world_orientations[n])
            < 1e-5
        )


def solve(args, points, **kwargs):
    return fit_connected_pose(
        **args,
        targets={
            n: LandmarkTarget(position=p, tolerance=5.0) for n, p in points.items()
        },
        rotation_tolerances_radians=dict.fromkeys(
            args["segment_relative_orientations"], 0.5
        ),
        **kwargs,
    )


def shoulders(landmarks):
    return {n: landmarks[n] for n in ("left_acromion", "right_acromion")}


def assert_geometry(args, result):
    for joint in args["skeleton"].joints.values():
        if joint.child.name in args["segment_names"]:
            np.testing.assert_allclose(
                result.world_origins[joint.child.name].array,
                result.landmarks[joint.connect_at.name].array,
                atol=1e-9,
            )
    for n in args["segment_names"]:
        primary = args["skeleton"].segments[n].frame_definition.primary_point_name
        assert np.linalg.norm(
            result.landmarks[primary].array - result.world_origins[n].array
        ) == pytest.approx(args["fit"].segment_lengths[n])
    np.testing.assert_array_equal(
        result.world_origins["pelvis"].array, args["root_origin"].array
    )
    assert result.world_orientations["pelvis"].is_same_rotation(
        other=args["root_world_orientation"]
    )


def test_existing_connected_pose_is_unchanged():
    args = setup()
    _, _, landmarks = synthesize_fitted_pose(**args)
    result = solve(args, shoulders(landmarks))
    assert result.termination == "stationary"
    assert result.final_cost < 1e-20
    assert result.targets_within_tolerance
    assert_geometry(args, result)


def test_two_shoulders_fit_together_without_stretching_or_mutating_inputs():
    args = setup()
    reference = dict(args["segment_relative_orientations"])
    changed = dict(reference)
    for n, vector in [
        ("sacrolumbar", [0.12, 0.08, 0]),
        ("thoracic", [-0.08, 0, 0.1]),
        ("left_clavicle", [0, 0.25, 0.08]),
        ("right_clavicle", [0, -0.2, -0.05]),
    ]:
        changed[n] = (
            RotationQuaternion.from_rotation_vector(rotation_vector=np.array(vector))
            * changed[n]
        )
    _, _, desired = synthesize_fitted_pose(
        **{**args, "segment_relative_orientations": changed}
    )
    result = solve(args, shoulders(desired))
    assert result.final_cost < result.initial_cost * 0.05
    assert result.targets_within_tolerance
    assert_geometry(args, result)
    assert all(
        args["segment_relative_orientations"][n].is_same_rotation(other=q)
        for n, q in reference.items()
    )


def test_unreachable_targets_report_residual_and_keep_geometry():
    args = setup()
    _, _, landmarks = synthesize_fitted_pose(**args)
    targets = {
        n: Point.from_array(values=p.array + np.array([0.0, 0.0, 3000.0]))
        for n, p in shoulders(landmarks).items()
    }
    result = solve(args, targets, max_iterations=12)
    assert not result.targets_within_tolerance
    assert result.final_cost <= result.initial_cost
    assert all(np.isfinite(list(result.target_errors.values())))
    assert_geometry(args, result)


def test_world_coordinate_change_does_not_change_local_solution():
    args = setup()
    _, _, landmarks = synthesize_fitted_pose(**args)
    targets = {
        n: Point.from_array(values=p.array + np.array([15.0, -10.0, 20.0]))
        for n, p in shoulders(landmarks).items()
    }
    first = solve(args, targets)
    q = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.4, -0.7, 0.3])
    )
    t = np.array([-100.0, 500.0, 25.0])
    moved = {
        **args,
        "root_origin": Point.from_array(
            values=q.rotate_vector(vector=args["root_origin"].array) + t
        ),
        "root_world_orientation": q * args["root_world_orientation"],
    }
    second = solve(
        moved,
        {
            n: Point.from_array(values=q.rotate_vector(vector=p.array) + t)
            for n, p in targets.items()
        },
    )
    for n in args["segment_relative_orientations"]:
        assert (
            first.relative_orientations[n].angle_to(
                other=second.relative_orientations[n]
            )
            < 1e-5
        )


def test_missing_target_can_be_omitted_but_not_invented():
    args = setup()
    _, _, landmarks = synthesize_fitted_pose(**args)
    result = solve(args, {"left_acromion": landmarks["left_acromion"]})
    assert set(result.target_errors) == {"left_acromion"}
    with pytest.raises(ValueError, match="one target"):
        solve(args, {})
    with pytest.raises(ValueError, match="finite point"):
        solve(
            args,
            {
                "left_acromion": Point.from_prevalidated_array(
                    array=np.array([np.nan, 0.0, 0.0])
                )
            },
        )


def test_small_target_jitter_does_not_cause_large_rotations():
    args = setup()
    _, _, landmarks = synthesize_fitted_pose(**args)
    rng = np.random.default_rng(3)
    for _ in range(6):
        targets = {
            n: Point.from_array(values=p.array + rng.uniform(-1, 1, 3))
            for n, p in shoulders(landmarks).items()
        }
        result = solve(args, targets)
        for n, q in args["segment_relative_orientations"].items():
            assert result.relative_orientations[n].angle_to(other=q) < np.deg2rad(2.0)
        assert_geometry(args, result)


def test_upper_body_static_noise_keeps_connections_and_bounded_rotations():
    args = setup()
    rest = RestPose.from_default_yaml(skeleton=args["skeleton"])
    names = args["segment_names"] | frozenset(
        (
            "cervical_spine",
            "skull",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_arm",
            "right_lower_arm",
        )
    )
    args = {
        **args,
        "segment_names": names,
        "segment_relative_orientations": {
            n: rest.relative_orientations[n] for n in names if n != "pelvis"
        },
    }
    _, _, exact = synthesize_fitted_pose(**args)
    rng = np.random.default_rng(42)
    target_names = (
        "left_hip_socket",
        "right_hip_socket",
        "left_acromion",
        "right_acromion",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_ear",
        "right_ear",
        "nose",
    )
    targets = {
        n: LandmarkTarget(
            Point.from_array(values=exact[n].array + rng.normal(0.0, 1.0, 3)), 1.0
        )
        for n in target_names
    }
    result = fit_connected_pose(
        **args,
        targets=targets,
        rotation_tolerances_radians=dict.fromkeys(
            args["segment_relative_orientations"], 0.5
        ),
        root_tolerances=RootPoseTolerances(1000.0, 1.0),
    )
    assert result.final_cost < result.initial_cost
    assert_geometry(
        {
            **args,
            "root_origin": result.world_origins["pelvis"],
            "root_world_orientation": result.world_orientations["pelvis"],
        },
        result,
    )
    # Regression bound for this explicit 1 mm noise/prior setup, not an
    # anatomical range or a guarantee for arbitrary noise and missing points.
    for name, reference in args["segment_relative_orientations"].items():
        assert result.relative_orientations[name].angle_to(
            other=reference
        ) < np.deg2rad(5.0)


def test_unobserved_twist_uses_explicit_pose_prior_not_initial_guess():
    args = setup()
    rest = RestPose.from_default_yaml(skeleton=args["skeleton"])
    names = args["segment_names"] | frozenset(("left_upper_arm", "left_lower_arm"))
    neutral = {n: rest.relative_orientations[n] for n in names if n != "pelvis"}
    args = {**args, "segment_names": names, "segment_relative_orientations": neutral}
    _, _, points = synthesize_fitted_pose(**args)
    initial = dict(neutral)
    initial["left_lower_arm"] = neutral[
        "left_lower_arm"
    ] * RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.0, 0.0, 0.8])
    )
    result = fit_connected_pose(
        **{**args, "segment_relative_orientations": initial},
        targets={"left_wrist": LandmarkTarget(points["left_wrist"], 1.0)},
        rotation_tolerances_radians={"left_lower_arm": 0.5},
        pose_prior_orientations=neutral,
    )
    assert result.target_errors["left_wrist"] < 1e-6
    assert (
        result.relative_orientations["left_lower_arm"].angle_to(
            other=neutral["left_lower_arm"]
        )
        < 1e-5
    )
