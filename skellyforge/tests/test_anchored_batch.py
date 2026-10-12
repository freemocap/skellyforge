"""Cross-frame synthesis preserves scalar geometry, selection, and chunk boundaries."""

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.anchored_batch import AnchoredSkeletonGeometry
from skellyforge.core.skeleton.chain.synthesis import synthesize_anchored_pose
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def batch_fixture(anchor="pelvis", frames=17):
    skeleton = SkeletonDefinition.from_default_yaml()
    scales = {name: 1400. + i * 11 for i, name in enumerate(skeleton.segments)}
    fit = ModelScaleFit(fitted_scale=1700., segment_scales=scales,
        segment_lengths={n: s.length * scales[n] for n, s in skeleton.segments.items()},
        measured_segment_names=frozenset(), voting_segment_names=frozenset())
    geometry = AnchoredSkeletonGeometry.from_skeleton(skeleton=skeleton, fit=fit, anchor_segment_name=anchor)
    rng = np.random.default_rng(27)
    rotations = rng.normal(size=(frames, len(geometry.segment_names), 4))
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    anchors = rng.normal(size=(frames, 3)) * 1000
    present = np.ones(rotations.shape[:2], dtype=bool)
    # Vary observed branches across frames, retaining anchor connectivity.
    for parents, children, _, _ in geometry.traversal_levels:
        present[:, children] = present[:, parents] & (rng.random((frames, len(children))) > .12)
    if frames > 2:
        present[0] = False
        present[1] = False
        present[1, geometry.anchor_index] = True
        present[2] = True
    rotations[~present] = np.nan
    return skeleton, fit, geometry, rotations, present, anchors


@pytest.mark.parametrize("anchor", ["pelvis", "skull", "left_carpals", "right_carpals"])
def test_batch_matches_scalar_for_full_partial_and_empty_frames(anchor):
    skeleton, fit, geometry, rotations, present, anchors = batch_fixture(anchor)
    before = rotations.copy()
    actual = geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors)
    for frame in range(len(rotations)):
        if not present[frame].any():
            assert np.isnan(actual.segment_origins[frame]).all()
            assert np.isnan(actual.landmark_positions[frame]).all()
            continue
        _, origins, landmarks = synthesize_anchored_pose(skeleton=skeleton, fit=fit,
            segment_world_orientations={name: RotationQuaternion(*map(float, rotations[frame, i]))
                for i, name in enumerate(geometry.segment_names) if present[frame, i]},
            anchor_segment_name=anchor, anchor_origin=Point.from_array(values=anchors[frame]))
        for i, name in enumerate(geometry.segment_names):
            assert actual.segment_present[frame, i] == (name in origins)
            if name in origins:
                np.testing.assert_array_equal(actual.segment_origins[frame, i], origins[name].array)
        for i, name in enumerate(geometry.landmark_names):
            assert actual.landmark_present[frame, i] == (name in landmarks)
            if name in landmarks:
                np.testing.assert_array_equal(actual.landmark_positions[frame, i], landmarks[name].array)
            else:
                assert np.isnan(actual.landmark_positions[frame, i]).all()
    np.testing.assert_array_equal(rotations, before)
    for chunk_size in [1, 4, 16]:
        chunks = [geometry.positions(world_orientations=rotations[i:i+chunk_size],
            segment_present=present[i:i+chunk_size], anchor_origins=anchors[i:i+chunk_size])
            for i in range(0, len(rotations), chunk_size)]
        np.testing.assert_array_equal(np.concatenate([c.landmark_positions for c in chunks]), actual.landmark_positions)


def test_batch_empty_and_invalid_inputs():
    _, _, geometry, rotations, present, anchors = batch_fixture()
    empty = geometry.positions(world_orientations=rotations[:0], segment_present=present[:0], anchor_origins=anchors[:0])
    assert empty.landmark_positions.shape == (0, len(geometry.landmark_names), 3)
    with pytest.raises(ValueError, match="boolean"):
        geometry.positions(world_orientations=rotations, segment_present=present.astype(int), anchor_origins=anchors)
    bad = rotations.copy()
    bad[2, geometry.anchor_index] *= 2
    with pytest.raises(ValueError, match="unit"):
        geometry.positions(world_orientations=bad, segment_present=present, anchor_origins=anchors)
    missing = present.copy()
    missing[2, geometry.anchor_index] = False
    with pytest.raises(ValueError, match="include the anchor"):
        geometry.positions(world_orientations=rotations, segment_present=missing, anchor_origins=anchors)
    missing[2] = False
    missing[2, geometry.anchor_index] = True
    missing[2, geometry.segment_names.index("left_carpals")] = True
    with pytest.raises(ValueError, match="without gaps"):
        geometry.positions(world_orientations=rotations, segment_present=missing, anchor_origins=anchors)


def test_prepared_geometry_does_not_alias_fit_or_model():
    _, fit, geometry, rotations, present, anchors = batch_fixture()
    expected = geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors)
    fit.segment_scales["pelvis"] *= 2
    actual = geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors)
    np.testing.assert_array_equal(actual.landmark_positions, expected.landmark_positions)
    assert not geometry.attachment_vectors.flags.writeable


def test_batch_rejects_bad_shapes_and_active_nonfinite_anchor():
    _, _, geometry, rotations, present, anchors = batch_fixture()
    with pytest.raises(ValueError, match="Orientations must have shape"):
        geometry.positions(world_orientations=rotations[..., :3], segment_present=present, anchor_origins=anchors)
    with pytest.raises(ValueError, match="Anchor origins must have shape"):
        geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors[:, :2])
    anchors[0] = np.nan  # Empty frames do not invent a position.
    geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors)
    anchors[2] = np.nan
    with pytest.raises(ValueError, match="Active anchor origins must be finite"):
        geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors)
