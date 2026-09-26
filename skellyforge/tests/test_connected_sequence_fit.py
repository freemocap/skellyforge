"""Time-aware connected fitting: geometry, missingness and explicit priors."""

from dataclasses import replace
import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    LandmarkTarget,
    RootPoseTolerances,
    fit_connected_pose,
)
from skellyforge.core.skeleton.pose.fit_connected_sequence import (
    ConnectedFrameInput,
    MotionTolerances,
    fit_connected_sequence,
)
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def fixture():
    skeleton = SkeletonDefinition.from_default_yaml()
    fit = ModelScaleFit(
        fitted_scale=1700.0,
        segment_scales=dict.fromkeys(skeleton.segments, 1700.0),
        segment_lengths={n: s.length * 1700 for n, s in skeleton.segments.items()},
        measured_segment_names=frozenset(),
        voting_segment_names=frozenset(),
    )
    pose = dict(
        skeleton=skeleton,
        fit=fit,
        segment_names=frozenset(("pelvis",)),
        segment_relative_orientations={},
        root_origin=Point.from_xyz(x=0.0, y=0.0, z=800.0),
        root_world_orientation=RotationQuaternion.identity(),
    )
    _, _, points = synthesize_fitted_pose(**pose)
    targets = {
        n: LandmarkTarget(points[n], 1.0)
        for n in ("left_hip_socket", "right_hip_socket", "sacrum_top")
    }
    frames = tuple(
        ConnectedFrameInput(
            t, targets, {}, pose["root_world_orientation"], pose["root_origin"]
        )
        for t in (0.0, 0.03, 0.07, 0.12)
    )
    settings = dict(
        skeleton=skeleton,
        fit=fit,
        segment_names=pose["segment_names"],
        rotation_tolerances_radians={},
        root_tolerances=RootPoseTolerances(1000.0, 1.0),
        motion_tolerances=MotionTolerances({}, 0.2, 100.0),
    )
    return pose, frames, settings


def test_exact_static_sequence_stays_exact_and_missing_frame_is_explicit():
    pose, frames, settings = fixture()
    frames = (*frames[:1], replace(frames[1], targets={}), *frames[2:])
    result = fit_connected_sequence(frames=frames, **settings)
    assert result.converged
    assert result.target_counts == (3, 0, 3, 3)
    assert not result.frames[1].targets_within_tolerance
    assert result.final_cost < 1e-20
    for frame in result.frames:
        np.testing.assert_array_equal(
            frame.world_origins["pelvis"].array, pose["root_origin"].array
        )


def test_joint_window_reduces_static_jitter_and_keeps_rigid_geometry():
    pose, frames, settings = fixture()
    rng = np.random.default_rng(7)
    frames = tuple(
        replace(
            f,
            targets={
                n: LandmarkTarget(
                    Point.from_array(values=t.position.array + rng.normal(0.0, 1.0, 3)),
                    1.0,
                )
                for n, t in f.targets.items()
            },
        )
        for f in frames
    )
    independent = [
        fit_connected_pose(
            **pose,
            targets=f.targets,
            rotation_tolerances_radians={},
            root_tolerances=settings["root_tolerances"],
        )
        for f in frames
    ]
    result = fit_connected_sequence(frames=frames, **settings)
    assert result.converged
    assert result.final_cost < result.initial_cost

    def variation(results):
        return sum(
            a.world_orientations["pelvis"].angle_to(
                other=b.world_orientations["pelvis"]
            )
            ** 2
            for a, b in zip(results[:-1], results[1:])
        )

    assert variation(result.frames) < variation(independent)
    for frame in result.frames:
        for n, t in frames[0].targets.items():
            assert np.linalg.norm(
                frame.landmarks[n].array - frame.world_origins["pelvis"].array
            ) == pytest.approx(
                np.linalg.norm(pose["skeleton"].landmarks[n].local_position.array)
                * 1700.0
            )


@pytest.mark.parametrize(
    "times", [(0.0, 0.0, 1.0, 2.0), (0.0, 2.0, 1.0, 3.0), (0.0, 1.0, np.nan, 3.0)]
)
def test_invalid_timestamps_fail(times):
    _, frames, settings = fixture()
    with pytest.raises(ValueError, match="timestamps"):
        fit_connected_sequence(
            frames=tuple(replace(f, time_seconds=t) for f, t in zip(frames, times)),
            **settings,
        )


def test_no_observations_rejected_and_budget_exhaustion_reported():
    _, frames, settings = fixture()
    with pytest.raises(ValueError, match="observation"):
        fit_connected_sequence(
            frames=tuple(replace(f, targets={}) for f in frames), **settings
        )
    frames = tuple(
        replace(
            f,
            root_origin=Point.from_array(values=f.root_origin.array + [10.0, 0.0, 0.0]),
        )
        for f in frames
    )
    result = fit_connected_sequence(frames=frames, max_evaluations=1, **settings)
    assert not result.converged
    assert result.evaluations == 1


def test_velocity_cost_uses_elapsed_time_not_frame_index():
    _, frames, settings = fixture()

    # Initial poses and observations translate together, so initial pose/point
    # costs are zero and the sole initial cost is the velocity integral.
    def moving(speed):
        return tuple(
            replace(
                f,
                root_origin=Point.from_array(
                    values=f.root_origin.array + [speed * f.time_seconds, 0, 0]
                ),
                targets={
                    n: LandmarkTarget(
                        Point.from_array(
                            values=t.position.array + [speed * f.time_seconds, 0, 0]
                        ),
                        t.tolerance,
                    )
                    for n, t in f.targets.items()
                },
            )
            for f in frames
        )

    result = fit_connected_sequence(frames=moving(50.0), max_evaluations=1, **settings)
    assert result.initial_cost == pytest.approx((50.0 / 100.0) ** 2 * 0.12)
    slower = tuple(replace(f, time_seconds=f.time_seconds * 2) for f in moving(50.0))
    second = fit_connected_sequence(frames=slower, max_evaluations=1, **settings)
    assert second.initial_cost == pytest.approx(result.initial_cost / 2)


def test_sequence_world_transform_and_unit_change_preserve_solution():
    pose, frames, settings = fixture()
    frames = tuple(
        replace(
            f,
            targets={
                n: LandmarkTarget(
                    Point.from_array(values=t.position.array + [2.0 * i, 1.0, -1.0]),
                    t.tolerance,
                )
                for n, t in f.targets.items()
            },
        )
        for i, f in enumerate(frames)
    )
    first = fit_connected_sequence(frames=frames, **settings)
    q = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.3, -0.2, 0.4])
    )
    unit = 0.001

    def transform(p):
        return Point.from_array(
            values=q.rotate_vector(vector=p.array) * unit + [0.4, -0.1, 0.2]
        )

    moved = tuple(
        replace(
            f,
            root_origin=transform(f.root_origin),
            root_orientation=q * f.root_orientation,
            targets={
                n: LandmarkTarget(transform(t.position), t.tolerance * unit)
                for n, t in f.targets.items()
            },
        )
        for f in frames
    )
    old_fit = settings["fit"]
    scaled = replace(
        old_fit,
        fitted_scale=old_fit.fitted_scale * unit,
        segment_scales={n: v * unit for n, v in old_fit.segment_scales.items()},
        segment_lengths={n: v * unit for n, v in old_fit.segment_lengths.items()},
    )
    second = fit_connected_sequence(
        frames=moved,
        **{
            **settings,
            "fit": scaled,
            "root_tolerances": RootPoseTolerances(1.0, 1.0),
            "motion_tolerances": MotionTolerances({}, 0.2, 0.1),
        },
    )
    assert first.converged and second.converged
    assert second.final_cost == pytest.approx(first.final_cost, rel=1e-5)
    for a, b in zip(first.frames, second.frames):
        np.testing.assert_allclose(
            b.world_origins["pelvis"].array,
            transform(a.world_origins["pelvis"]).array,
            atol=1e-6,
        )
        assert (
            b.world_orientations["pelvis"].angle_to(
                other=q * a.world_orientations["pelvis"]
            )
            < 1e-4
        )


def test_moving_observations_are_not_frozen_by_velocity_prior():
    _, frames, settings = fixture()
    frames = tuple(
        replace(
            f,
            targets={
                n: LandmarkTarget(
                    Point.from_array(
                        values=t.position.array + [50.0 * f.time_seconds, 0.0, 0.0]
                    ),
                    t.tolerance,
                )
                for n, t in f.targets.items()
            },
        )
        for f in frames
    )
    result = fit_connected_sequence(frames=frames, **settings)
    assert result.converged
    assert max(error for f in result.frames for error in f.target_errors.values()) < 0.5
    travel = (
        result.frames[-1].world_origins["pelvis"].array[0]
        - result.frames[0].world_origins["pelvis"].array[0]
    )
    assert travel > 5.0  # True travel is 6 mm; this is a declared test-case bound.


def test_quaternion_sign_is_not_treated_as_motion():
    _, frames, settings = fixture()
    frames = tuple(
        (
            replace(f, root_orientation=RotationQuaternion(w=-1.0, x=0.0, y=0.0, z=0.0))
            if i % 2
            else f
        )
        for i, f in enumerate(frames)
    )
    result = fit_connected_sequence(frames=frames, **settings)
    assert result.initial_cost < 1e-20
    assert result.final_cost < 1e-20
