import json

import numpy as np
import pytest

from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
)
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.stl_rigid_body import (
    build_stl_rigid_body,
    load_stl_rigid_body,
)

POSITIONS = np.array(
    [[10, 20, 30], [40, 25, 35], [20, 55, 31], [15, 30, 65], [31, 42, 53]], dtype=float
)
NAMES = tuple(f"feature_{index}" for index in range(len(POSITIONS)))


def make_definition(**changes):
    arguments = dict(
        name="test_stl",
        feature_names=NAMES,
        object_positions_mm=POSITIONS,
        reference_unit_mm=60.0,
        source_mesh_to_object_mm=np.eye(4),
        source_mesh_units="mm",
    )
    arguments.update(changes)
    return build_stl_rigid_body(**arguments)


def measured_pose(definition, positions):
    observed = {
        name: Point.from_array(values=position)
        for name, position in zip(NAMES, positions)
    }
    return hydrate_skeleton(
        skeleton=definition.skeleton, observed=observed, require_all=True
    ).segment_poses["rigid_object"]


def mesh_world(definition, pose):
    transform = Transform(
        rotation=pose.orientation,
        translation=Displacement.from_array(values=pose.origin.array),
    )
    return definition.source_mesh_to_world_matrix(
        segment_to_world=transform, scale_mm=pose.scale_estimate
    )


def test_source_origin_and_rotation_recovered_through_existing_hydration():
    definition = make_definition()
    rotation = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.4, -0.2, 0.3])
    ).to_rotation_matrix()
    translation = np.array([230, -470, 810])
    world = POSITIONS @ rotation.T + translation
    pose = measured_pose(definition, world)
    assert pose.scale_estimate == pytest.approx(60)
    result = mesh_world(definition, pose)
    np.testing.assert_allclose(result[:3, :3], rotation, atol=1e-10)
    np.testing.assert_allclose(result[:3, 3], translation, atol=1e-10)
    np.testing.assert_allclose(
        definition.source_mesh_to_segment_reference
        @ definition.segment_reference_to_source_mesh,
        np.eye(4),
        atol=1e-12,
    )


def test_authored_segment_axes_are_consistent_and_all_features_remain_fit_points():
    definition = make_definition()
    segment = definition.skeleton.segments["rigid_object"]
    basis = calculate_orthonormal_basis(
        points={name: item.local_position for name, item in segment.landmarks.items()},
        definition=segment.frame_definition,
    )
    np.testing.assert_allclose(basis.world_from_local_matrix, np.eye(3), atol=1e-12)
    assert segment.rigid_point_set.point_names == NAMES
    triad = set(segment.frame_definition.point_names)
    extra = next(index for index, name in enumerate(NAMES) if name not in triad)
    changed = POSITIONS.copy()
    changed[extra] += [4, -2, 3]
    pose = measured_pose(definition, changed)
    assert not np.allclose(mesh_world(definition, pose), np.eye(4))


def test_missing_reference_marker_does_not_prevent_rigid_fit():
    definition = make_definition()
    segment = definition.skeleton.segments["rigid_object"]
    observed = {
        name: Point.from_array(values=position)
        for name, position in zip(NAMES, POSITIONS)
        if name != segment.frame_definition.origin_point_name
    }
    pose = hydrate_skeleton(
        skeleton=definition.skeleton, observed=observed, require_all=True
    ).segment_poses["rigid_object"]
    np.testing.assert_allclose(mesh_world(definition, pose), np.eye(4), atol=1e-10)


def test_source_units_normalization_and_original_origin_are_retained():
    transform = np.eye(4)
    transform[:3, :3] *= 1000
    transform[:3, 3] = [-400, 100, 80]
    definition = make_definition(
        source_mesh_to_object_mm=transform, source_mesh_units="m"
    )
    pose = measured_pose(definition, POSITIONS)
    np.testing.assert_allclose(mesh_world(definition, pose), transform, atol=1e-10)
    assert not definition.source_mesh_to_segment_reference.flags.writeable
    assert definition.source_mesh_units == "m"


def test_planar_geometry_is_valid_but_collinear_or_nonfinite_geometry_is_not():
    flat = POSITIONS.copy()
    flat[:, 2] = 5
    assert (
        make_definition(object_positions_mm=flat)
        .skeleton.segments["rigid_object"]
        .supports_rigid_fit
    )
    line = np.column_stack((np.arange(5), np.zeros((5, 2))))
    with pytest.raises(ValueError):
        make_definition(object_positions_mm=line)
    invalid = POSITIONS.copy()
    invalid[2, 0] = np.nan
    with pytest.raises(ValueError):
        make_definition(object_positions_mm=invalid)


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_names": ("same",) * 5},
        {"reference_unit_mm": 0.0},
        {"source_mesh_units": "unspecified"},
        {"source_mesh_to_object_mm": np.diag([-1, 1, 1, 1])},
        {"source_mesh_to_object_mm": np.diag([2, 1, 1, 1])},
    ],
)
def test_invalid_contract_rejected(changes):
    with pytest.raises(ValueError):
        make_definition(**changes)


def test_generated_and_retained_authored_files_are_equivalent(tmp_path):
    rotation = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0, 0, 0.7])
    ).to_rotation_matrix()
    origin = np.array([8, -5, 3])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    common = dict(
        units="mm",
        feature_names=NAMES,
        object_positions_mm=POSITIONS.tolist(),
        reference_unit_mm=60,
    )
    authored = {
        **common,
        "source_to_object_rotation": rotation.tolist(),
        "origin_source_mm": origin.tolist(),
        "source_mesh_sha256": "fixture",
    }
    generated = {
        **common,
        "schema_version": 1,
        "source_mesh_to_object": transform.tolist(),
        "original_mesh_units": "mm",
        "original_mesh_sha256": "fixture",
    }
    definitions = []
    for index, data in enumerate((authored, generated)):
        path = tmp_path / f"definition_{index}.json"
        path.write_text(json.dumps(data))
        definitions.append(load_stl_rigid_body(path=path, name="fixture"))
    np.testing.assert_allclose(
        definitions[0].source_mesh_to_segment_reference,
        definitions[1].source_mesh_to_segment_reference,
    )
    assert definitions[0].feature_names == NAMES
    assert definitions[1].mesh_sha256 == "fixture"
    generated["schema_version"] = 2
    path.write_text(json.dumps(generated))
    with pytest.raises(ValueError, match="schema_version"):
        load_stl_rigid_body(path=path, name="fixture")
