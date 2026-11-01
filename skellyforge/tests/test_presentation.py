"""Presentation is reproducible, finite, and follows the actual hydrated local frame."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_segment
from skellyforge.core.skeleton.presentation import Presentation, SuperquadricContours, compile_contours
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.core.skeleton.pose.rest_pose import RestPose


def recipe():
    return SuperquadricContours(segment='skull', crown='head_vertex', chin='chin',
        left_ear='left_ear', right_ear='right_ear', left_eye='left_eye', right_eye='right_eye')


@pytest.mark.parametrize('target', ['nose', 'left_eye'])
def test_ear_section_intersects_ears_and_faces_target_without_deforming_surface(target):
    skeleton = SkeletonDefinition.from_default_yaml()
    chosen = replace(skeleton.presentation.contours[0], nose=target)
    segment = skeleton.segments['skull']
    compiled = compile_contours(recipe=chosen, segment=segment)
    ring = np.array(compiled.positions).reshape(5, 64, 2, 3)[2, :, 0]
    left, right = (segment.landmarks[n].local_position.array for n in ('left_ear', 'right_ear'))
    np.testing.assert_allclose(ring[[0, 32]], [right, left], atol=1e-12)
    origin = (left + right) / 2
    lateral = (right - left) / np.linalg.norm(right - left)
    direction = segment.landmarks[target].local_position.array - origin
    direction -= lateral * np.dot(direction, lateral)
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose((ring[16] - origin) / np.linalg.norm(ring[16] - origin), direction, atol=1e-12)
    normal = np.cross(lateral, direction)
    np.testing.assert_allclose((ring - origin) @ normal, 0, atol=1e-12)
    assert compiled.surfaces == skeleton.presentation_lines[0].surfaces


def test_surface_materials_round_trip_and_legacy_recipe_has_no_new_surface():
    skeleton = SkeletonDefinition.from_default_yaml()
    saved = json.loads(json.dumps(asdict(SkeletonSnapshot.capture(skeleton)), default=lambda value: value.value))
    restored = SkeletonSnapshot.from_dict(saved).restore()
    assert restored.presentation_lines == skeleton.presentation_lines
    lines = restored.presentation_lines[0]
    assert lines.material.color == '#73747a' and lines.material.opacity == .75
    assert [(s.material.color, s.material.opacity) for s in lines.surfaces] == [('#444b61', .21), ('#5f6063', .25)]
    for surface in lines.surfaces:
        assert np.isfinite(surface.positions).all()
        assert max(surface.indices) < len(surface.positions) // 3
        assert len(surface.positions) // 3 < 600
    legacy = compile_contours(recipe=recipe(), segment=skeleton.segments['skull'])
    assert legacy.surfaces == () and legacy.material is None


@pytest.mark.parametrize('color,opacity', [('green', .5), ('#abcdef', -1.0), ('#abcdef', float('nan'))])
def test_invalid_materials_fail_at_definition_time(color, opacity):
    from skellyforge.core.skeleton.presentation import PresentationMaterial
    with pytest.raises(ValueError):
        PresentationMaterial(color=color, opacity=opacity)


def test_snapshot_and_legacy_are_independent_of_current_defaults():
    skeleton = replace(SkeletonDefinition.from_default_yaml(), presentation=Presentation(contours=(recipe(),)))
    saved = json.loads(json.dumps(asdict(SkeletonSnapshot.capture(skeleton)), default=lambda value: value.value))
    restored = SkeletonSnapshot.from_dict(saved).restore()
    assert restored.presentation == skeleton.presentation
    assert restored.presentation_lines == skeleton.presentation_lines
    saved.pop('presentation')
    legacy = SkeletonSnapshot.from_dict(saved).restore()
    assert legacy.presentation is None
    assert legacy.presentation_lines == ()


def test_closed_finite_contours_follow_hydrated_pose():
    segment = SkeletonDefinition.from_default_yaml().segments['skull']
    lines = compile_contours(recipe=recipe(), segment=segment)
    edges = np.array(lines.positions).reshape(5, 64, 2, 3)
    np.testing.assert_array_equal(edges[:, :, 1], np.roll(edges[:, :, 0], -1, axis=1))
    assert np.isfinite(edges).all()
    rotation = Rotation.from_euler('xyz', [.3, -.6, .8]).as_matrix()
    origin, scale = np.array([120., -80., 600.]), 1720.
    observed = {name: Point.from_array(values=origin + scale * (rotation @ landmark.local_position.array))
                for name, landmark in segment.landmarks.items()}
    pose = hydrate_segment(segment=segment, observed=observed)
    np.testing.assert_allclose(pose.origin.array, origin, atol=1e-9)
    assert pose.scale_estimate == pytest.approx(scale)
    np.testing.assert_allclose(pose.orientation.to_rotation_matrix(), rotation, atol=1e-10)
    # Renderer scale contract uses length / authored length, without rest rotation.
    local = edges.reshape(-1, 3)
    np.testing.assert_allclose(origin + (local @ rotation.T) * scale,
        pose.origin.array + (local @ pose.orientation.to_rotation_matrix().T) * (segment.length * pose.scale_estimate / lines.authored_length))


def test_selected_surface_contains_ears_eyes_and_crown_and_all_five_sections():
    skeleton = SkeletonDefinition.from_default_yaml()
    segment = skeleton.segments['skull']
    chosen = skeleton.presentation.contours[0]
    compiled = compile_contours(recipe=chosen, segment=segment)
    vertices = np.concatenate([np.array(compiled.positions).reshape(-1, 3),
        *(np.array(surface.positions).reshape(-1, 3) for surface in compiled.surfaces)])
    a = vertices[:, 0].max()
    b = vertices[:, 1].max()
    cz = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    c = (vertices[:, 2].max() - vertices[:, 2].min()) / 2
    landmarks = np.array([segment.landmarks[n].local_position.array for n in
                          ('head_vertex', 'left_ear', 'right_ear', 'left_eye', 'right_eye')])
    points = np.concatenate((vertices, landmarks))
    depth = np.where(points[:, 1] >= 0, b, b * chosen.posterior_depth)
    implicit = ((np.abs(points[:, 0] / a) ** (2 / chosen.horizontal_exponent)
                + np.abs(points[:, 1] / depth) ** (2 / chosen.horizontal_exponent)) ** (chosen.horizontal_exponent / chosen.vertical_exponent)
                + np.abs((points[:, 2] - cz) / c) ** (2 / chosen.vertical_exponent))
    np.testing.assert_allclose(implicit, 1.0, atol=1e-12)


@pytest.mark.parametrize('field,value', [('samples', 0), ('vertical_exponent', float('nan')), ('posterior_depth', -1.0)])
def test_invalid_parameters_fail_at_definition_time(field, value):
    with pytest.raises(ValueError):
        compile_contours(recipe=replace(recipe(), **{field: value}), segment=SkeletonDefinition.from_default_yaml().segments['skull'])


def test_missing_references_fail_at_definition_time():
    with pytest.raises(KeyError):
        replace(SkeletonDefinition.from_default_yaml(), presentation=Presentation(contours=(replace(recipe(), crown='missing'),)))


def test_selected_connections_and_sc_rest_propagation_are_explicit():
    current = SkeletonDefinition.from_default_yaml()
    snapshot = SkeletonSnapshot.capture(current)
    old = replace(snapshot, presentation=None, landmarks=tuple(
        replace(item, position=(item.position[0], 0.048593, item.position[2]))
        if item.name in ('left_sternoclavicular', 'right_sternoclavicular') else item
        for item in snapshot.landmarks)).restore()
    before, after = RestPose.from_default_yaml(skeleton=old), RestPose.from_default_yaml(skeleton=current)
    assert current.joints.keys() == old.joints.keys()
    assert current.landmarks.keys() == old.landmarks.keys()
    pelvis_pairs = current.landmark_connections['pelvis_rim_and_struts'].pairs
    assert len(pelvis_pairs) == 22
    assert len(current.landmark_connections['thorax_bridge'].pairs) == 6
    assert ('chest_center', 'sacrum_top') in current.landmark_connections['thorax_bridge'].pairs
    assert current.landmarks['xiphoid_process'].local_position.array[1] == pytest.approx(.078964 * .6)
    np.testing.assert_allclose(current.landmarks['sternoclavicular_notch'].local_position.array,
        (current.landmarks['left_sternoclavicular'].local_position.array
         + current.landmarks['right_sternoclavicular'].local_position.array) / 2, atol=1e-12)
    for side in ('left', 'right'):
        assert ('pelvis_origin', f'{side}_hip_socket') in pelvis_pairs
        assert (f'{side}_sacroiliac_joint', 'sacrum_top') in pelvis_pairs
        assert (f'{side}_sacroiliac_joint', 'coccyx_tip') in pelvis_pairs
        assert current.landmarks[f'{side}_sternoclavicular'].local_position.array[1] == pytest.approx(0.048593 * .7)
        for point in ('sternoclavicular', 'acromion', 'elbow', 'wrist'):
            np.testing.assert_allclose(after.landmark_positions[f'{side}_{point}'].array - before.landmark_positions[f'{side}_{point}'].array,
                                       [0, -0.0145779, 0], atol=1e-12)
    for point in ('head_vertex', 'xiphoid_process', 'neck_center', 'left_hip_socket'):
        np.testing.assert_array_equal(after.landmark_positions[point].array, before.landmark_positions[point].array)
    assert ('thoracic', 'left_clavicle') in current.presentation.excluded_origin_edges
    assert current.presentation.excluded_bones == ('skull', 'pelvis')


def test_xiphoid_depth_change_preserves_other_rest_landmarks():
    current = SkeletonDefinition.from_default_yaml()
    snapshot = SkeletonSnapshot.capture(current)
    old = replace(snapshot, landmarks=tuple(
        replace(item, position=(item.position[0], .078964, item.position[2]))
        if item.name == 'xiphoid_process' else item for item in snapshot.landmarks)).restore()
    before = RestPose.from_default_yaml(skeleton=old)
    after = RestPose.from_default_yaml(skeleton=current)
    for name, point in after.landmark_positions.items():
        expected = [0, -.078964 * .4, 0] if name == 'xiphoid_process' else [0, 0, 0]
        np.testing.assert_allclose(point.array - before.landmark_positions[name].array, expected, atol=1e-12)
