"""End-to-end tests for the biomechanics pipeline, in the T-pose."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.biomechanics.anthropometric_parameters import AnthropometricParameters
from skellyforge.core.biomechanics.center_of_mass import (
    CenterOfMassDefinitions,
    compute_segment_coms,
)
from skellyforge.core.biomechanics.composite_inertia import (
    body_inertial_properties,
    whole_body_center_of_mass,
    whole_body_inertia_tensor,
)
from skellyforge.core.biomechanics.derived_kinematics import (
    center_of_mass_acceleration,
    center_of_mass_velocity,
)
from skellyforge.core.biomechanics.ground_reference import (
    center_of_pressure,
    centroidal_moment_pivot,
    extrapolated_center_of_mass,
)
from skellyforge.core.biomechanics.segment_mapping import distribute_segment_masses
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import (
    PoseSolution,
    SegmentPose,
    SkeletonPose,
)
from skellyforge.type_overloads import FloatArray

SKELETON_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)
REST_POSE_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "rest_pose.yaml"
)

BODY_MASS: float = 70.0
GRAVITY: FloatArray = np.array([0.0, 0.0, -9810.0])
# The rest pose is authored as body-height proportions, and gravity above is in mm/s^2, so
# the T-pose these tests roll up is a real 1700mm body rather than a template of height 1.
# Every position here is therefore in millimetres, as it is in the live pipeline.
SUBJECT_HEIGHT_MM: float = 1700.0


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)


def _pose() -> SkeletonPose:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    return SkeletonPose(
        segment_poses={
            name: SegmentPose(
                segment_name=name,
                origin=Point.from_array(
                    values=SUBJECT_HEIGHT_MM * rest_pose.segment_origins[name].array
                ),
                orientation=rest_pose.segment_orientations[name],
                scale_estimate=SUBJECT_HEIGHT_MM,
                solved_by=PoseSolution.RIGID_FIT,
            )
            for name in skeleton.segments
        }
    )


def _segment_scales() -> dict[str, float]:
    """A perfectly-fitted subject: every segment at the same body height."""
    return {name: SUBJECT_HEIGHT_MM for name in _skeleton().segments}


def _anthropometric() -> AnthropometricParameters:
    return AnthropometricParameters.from_default_yaml()


def _com_definitions() -> CenterOfMassDefinitions:
    return CenterOfMassDefinitions.from_default_yaml()


def _body() -> FloatArray:
    return body_inertial_properties(
        skeleton=_skeleton(),
        pose=_pose(),
        body_mass=BODY_MASS,
        anthropometric=_anthropometric(),
        com_definitions=_com_definitions(),
        segment_scales=_segment_scales(),
    )


def test_distributed_masses_sum_to_the_body_mass() -> None:
    skeleton = _skeleton()
    masses = distribute_segment_masses(
        skeleton=skeleton, body_mass=BODY_MASS, anthropometric=_anthropometric()
    )
    assert len(masses) == len(skeleton.segments)
    assert sum(masses.values()) == pytest.approx(BODY_MASS, abs=1e-9)


def test_whole_body_mass_is_the_body_mass() -> None:
    assert _body().mass == pytest.approx(BODY_MASS)


def test_whole_body_com_is_on_the_midline_in_the_t_pose() -> None:
    assert _body().center_of_mass[0] == pytest.approx(0.0, abs=1e-9)


def test_whole_body_inertia_is_symmetric_and_positive_definite() -> None:
    tensor = _body().inertia_tensor
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(tensor) > 0.0)


def test_center_of_pressure_formula() -> None:
    force = np.array([0.0, 0.0, 1000.0])
    moment = np.array([20.0, -30.0, 0.0])
    cop = center_of_pressure(force=force, moment=moment)
    np.testing.assert_allclose(cop, np.array([0.03, 0.02, 0.0]), atol=1e-12)


def test_extrapolated_com_equals_com_ground_projection_when_still() -> None:
    com = _body().center_of_mass
    xcom = extrapolated_center_of_mass(
        com=com, com_velocity=np.zeros(3), gravity=GRAVITY
    )
    np.testing.assert_allclose(xcom, np.array([com[0], com[1], 0.0]), atol=1e-9)


def test_centroidal_moment_pivot_equals_com_ground_projection_without_horizontal_force() -> None:
    com = _body().center_of_mass
    force = np.array([0.0, 0.0, BODY_MASS * 9810.0])
    cmp_point = centroidal_moment_pivot(force=force, center_of_mass=com)
    np.testing.assert_allclose(cmp_point, np.array([com[0], com[1], 0.0]), atol=1e-9)


def test_center_of_mass_velocity_of_constant_motion() -> None:
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    positions = np.column_stack(
        [5.0 * timestamps, -2.0 * timestamps, 3.0 * timestamps]
    )
    velocity = center_of_mass_velocity(positions=positions, timestamps=timestamps)
    np.testing.assert_allclose(velocity, np.tile(np.array([5.0, -2.0, 3.0]), (4, 1)))


def test_center_of_mass_acceleration_of_constant_acceleration() -> None:
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    velocities = np.column_stack([timestamps, 2.0 * timestamps, -timestamps])
    acceleration = center_of_mass_acceleration(
        velocities=velocities, timestamps=timestamps
    )
    np.testing.assert_allclose(
        acceleration, np.tile(np.array([1.0, 2.0, -1.0]), (4, 1))
    )


def test_kinematics_rejects_non_increasing_timestamps() -> None:
    timestamps = np.array([0.0, 1.0, 1.0, 2.0])
    positions = np.zeros((4, 3))
    with pytest.raises(ValueError):
        center_of_mass_velocity(positions=positions, timestamps=timestamps)


def test_partial_pose_rolls_up_over_the_visible_segments() -> None:
    skeleton = _skeleton()
    dropped_name = sorted(skeleton.segments)[len(skeleton.segments) // 2]
    full_pose = _pose()
    partial_pose = SkeletonPose(
        segment_poses={
            name: segment_pose
            for name, segment_pose in full_pose.segment_poses.items()
            if name != dropped_name
        }
    )
    body = body_inertial_properties(
        skeleton=skeleton,
        pose=partial_pose,
        body_mass=BODY_MASS,
        anthropometric=_anthropometric(),
        com_definitions=_com_definitions(),
        segment_scales=_segment_scales(),
    )
    assert np.all(np.isfinite(body.center_of_mass))
    assert np.all(np.isfinite(body.inertia_tensor))


def test_occluded_com_landmarks_skip_their_segment_instead_of_crashing() -> None:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    world = {
        name: position.array.copy()
        for name, position in rest_pose.landmark_positions.items()
    }
    definitions = _com_definitions()

    # Remove every COM-defining landmark of one midline segment while leaving its
    # anchors (and everything else) in place - the exact shape of occlusion that
    # used to crash the inertia roll-up with a TypeError on a None COM.
    target = next(
        definition
        for definition in definitions.definitions.values()
        if not definition.sided
        and {entry.landmark for entry in definition.weights}.isdisjoint(
            {definition.proximal, definition.distal}
        )
    )
    for entry in target.weights:
        del world[entry.landmark]

    segment_coms = compute_segment_coms(definitions=definitions, world=world)
    assert target.name not in segment_coms
    assert whole_body_center_of_mass(
        segment_coms=segment_coms,
        segment_masses={
            name: BODY_MASS * 0.01 for name in definitions.all_segment_names
        },
    ) is not None


def test_whole_body_inertia_tensor_refuses_an_inertia_without_a_com() -> None:
    with pytest.raises(KeyError, match="no matching COM/mass"):
        whole_body_inertia_tensor(
            segment_inertias={"orphan_segment": np.eye(3)},
            segment_coms={},
            segment_masses={},
            body_center_of_mass=np.zeros(3),
        )
