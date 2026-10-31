"""Controlled weighted translation objective; no tracking-package dependencies."""
from dataclasses import replace

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import hamilton_product, rotate_vector_pairs
from skellyforge.core.skeleton.chain.weighted_attachments import WeightedAttachmentGeometry
from skellyforge.tests.test_anchored_batch import batch_fixture


def fixture(strength=1., anchor="pelvis", frames=17):
    skeleton, _, geometry, rotations, present, anchors = batch_fixture(anchor=anchor, frames=frames)
    observed = np.random.default_rng(41).normal(size=(*present.shape, 3)) * 500
    observed[:, geometry.anchor_index] = anchors
    observed[~present] = np.nan
    solver = WeightedAttachmentGeometry.from_skeleton(skeleton=skeleton,
        anchor_segment_name=anchor, strength=strength)
    kwargs = dict(geometry=geometry, world_orientations=rotations,
                  segment_present=present, observed_origins=observed)
    return solver, kwargs


def edge_offsets(kwargs):
    geometry = kwargs["geometry"]
    rotations = kwargs["world_orientations"]
    return rotate_vector_pairs(
        quaternions=rotations[:, geometry.attachment_owners].reshape((-1, 4)),
        vectors=np.broadcast_to(geometry.attachment_vectors,
            (len(rotations), len(geometry.attachment_owners), 3)).reshape((-1, 3))
    ).reshape((len(rotations), -1, 3))


@pytest.mark.parametrize("anchor", ["pelvis", "skull", "left_carpals"])
def test_zero_is_independent_and_large_strength_reaches_connected_limit(anchor):
    solver, kwargs = fixture(strength=0., anchor=anchor)
    actual = solver.positions(**kwargs)
    np.testing.assert_array_equal(actual.segment_origins, kwargs["observed_origins"])
    assert not solver._operators
    connected = kwargs["geometry"].positions(world_orientations=kwargs["world_orientations"],
        segment_present=kwargs["segment_present"], anchor_origins=kwargs["observed_origins"][:, solver.anchor_index])
    strong = replace(solver, strength=1e300).positions(**kwargs)
    np.testing.assert_allclose(strong.segment_origins, connected.segment_origins, atol=2e-9)


def test_monotonic_gaps_and_normal_equations_match_direct_constrained_solve():
    solver, kwargs = fixture()
    present, observed = kwargs["segment_present"], kwargs["observed_origins"]
    offsets = edge_offsets(kwargs)
    active_edges = present[:, solver.edge_parents] & present[:, solver.edge_children]
    residuals = []
    for strength in (0., .01, .1, 1., 10., 1000.):
        current = replace(solver, strength=strength)
        result = current.positions(**kwargs)
        gaps = result.segment_origins[:, solver.edge_children] - result.segment_origins[:, solver.edge_parents] - offsets
        residuals.append(np.sum(gaps[active_edges] ** 2))
        for frame in range(len(present)):
            free = np.flatnonzero(present[frame] & (np.arange(present.shape[1]) != solver.anchor_index))
            edges = np.flatnonzero(active_edges[frame])
            b = (solver.edge_children[edges, None] == free[None, :]).astype(float)
            b -= solver.edge_parents[edges, None] == free[None, :]
            error = observed[frame, solver.edge_children[edges]] - observed[frame, solver.edge_parents[edges]] - offsets[frame, edges]
            delta = np.linalg.solve(np.eye(len(free)) + strength * b.T @ b, -strength * b.T @ error)
            np.testing.assert_allclose(result.segment_origins[frame, free], observed[frame, free] + delta, atol=1e-8)
    assert np.all(np.diff(residuals) < 0)


def test_consistent_observations_are_fixed_and_landmark_offsets_keep_fitted_lengths():
    solver, kwargs = fixture()
    geometry = kwargs["geometry"]
    connected = geometry.positions(world_orientations=kwargs["world_orientations"],
        segment_present=kwargs["segment_present"], anchor_origins=kwargs["observed_origins"][:, solver.anchor_index])
    consistent = solver.positions(**dict(kwargs, observed_origins=connected.segment_origins))
    np.testing.assert_allclose(consistent.segment_origins, connected.segment_origins, atol=1e-10)
    result = solver.positions(**kwargs)
    offsets = result.landmark_positions - result.segment_origins[:, geometry.landmark_owners]
    expected = np.broadcast_to(np.linalg.norm(geometry.landmark_vectors, axis=-1), result.landmark_present.shape)
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=-1)[result.landmark_present], expected[result.landmark_present], atol=1e-10)


def test_missingness_disconnected_components_empty_frames_and_input_ownership():
    solver, kwargs = fixture()
    present = kwargs["segment_present"]
    present[2, solver.anchor_index] = False
    kwargs["world_orientations"][~present] = np.nan
    kwargs["observed_origins"][~present] = np.nan
    before = {name: values.copy() for name, values in kwargs.items() if isinstance(values, np.ndarray)}
    result = solver.positions(**kwargs)
    np.testing.assert_array_equal(result.segment_present, present)
    assert np.isnan(result.segment_origins[~present]).all()
    assert np.isnan(result.landmark_positions[~result.landmark_present]).all()
    assert np.isfinite(result.segment_origins[present]).all()
    for name, values in before.items():
        np.testing.assert_array_equal(kwargs[name], values)
        for output in (result.segment_origins, result.landmark_positions, result.segment_present, result.landmark_present):
            assert not np.shares_memory(kwargs[name], output)
    empty = solver.positions(**{name: value[:0] if isinstance(value, np.ndarray) else value for name, value in kwargs.items()})
    assert empty.segment_origins.shape == (0, present.shape[1], 3)


def test_rigid_transform_equivariance_and_batch_single_parity():
    solver, kwargs = fixture(anchor="right_carpals")
    result = solver.positions(**kwargs)
    q = np.array([.5, -.5, .5, -.5])
    shift = np.array([91., -83., 17.])
    def rotate(values):
        return rotate_vector_pairs(quaternions=np.broadcast_to(q, (values.size // 3, 4)),
                                   vectors=values.reshape((-1, 3))).reshape(values.shape)
    rotations = kwargs["world_orientations"]
    transformed_rotations = hamilton_product(left=np.broadcast_to(q, (rotations.size // 4, 4)),
        right=rotations.reshape((-1, 4))).reshape(rotations.shape)
    transformed = solver.positions(**dict(kwargs, observed_origins=rotate(kwargs["observed_origins"]) + shift,
                                          world_orientations=transformed_rotations))
    np.testing.assert_allclose(transformed.segment_origins, rotate(result.segment_origins) + shift, atol=1e-9)
    np.testing.assert_allclose(transformed.landmark_positions, rotate(result.landmark_positions) + shift, atol=1e-9)
    singles = [solver.positions(**{name: value[i:i+1] if isinstance(value, np.ndarray) else value
               for name, value in kwargs.items()}) for i in range(len(rotations))]
    np.testing.assert_allclose(np.concatenate([s.segment_origins for s in singles]), result.segment_origins, atol=1e-10)
    np.testing.assert_allclose(np.concatenate([s.landmark_positions for s in singles]), result.landmark_positions, atol=1e-10)


@pytest.mark.parametrize("strength", [-1., float("inf"), float("-inf"), float("nan")])
def test_invalid_strength(strength):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fixture(strength=strength)
    solver, _ = fixture()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(solver, strength=strength)


def test_input_validation():
    solver, kwargs = fixture()
    with pytest.raises(ValueError, match="boolean"):
        solver.positions(**dict(kwargs, segment_present=kwargs["segment_present"].astype(int)))
    with pytest.raises(ValueError, match="origins must have shape"):
        solver.positions(**dict(kwargs, observed_origins=kwargs["observed_origins"][..., :2]))
    bad = kwargs["observed_origins"].copy()
    bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="origins must be finite"):
        solver.positions(**dict(kwargs, observed_origins=bad))
    bad = kwargs["world_orientations"].copy()
    bad[2, 0] *= 2
    with pytest.raises(ValueError, match="unit quaternions"):
        solver.positions(**dict(kwargs, world_orientations=bad))
    geometry = replace(kwargs["geometry"], landmark_vectors=kwargs["geometry"].landmark_vectors * np.nan)
    with pytest.raises(ValueError, match="geometry vectors must be finite"):
        solver.positions(**dict(kwargs, geometry=geometry))


def test_replacing_warmed_solver_owns_fresh_cache_for_changed_strength():
    solver, kwargs = fixture(strength=1.)
    original = solver.positions(**kwargs)
    assert solver._operators
    changed = replace(solver, strength=100.)
    assert changed._operators is not solver._operators
    assert not changed._operators
    assert changed._validated_topology is None
    actual = changed.positions(**kwargs)
    fresh, _ = fixture(strength=100.)
    expected = fresh.positions(**kwargs)
    np.testing.assert_allclose(actual.segment_origins, expected.segment_origins)
    assert not np.allclose(actual.segment_origins, original.segment_origins, equal_nan=True)
    np.testing.assert_allclose(solver.positions(**kwargs).segment_origins, original.segment_origins)


def test_readonly_topology_validation_is_reused_but_fit_vectors_are_checked(monkeypatch):
    solver, kwargs = fixture()
    solver.positions(**kwargs)
    geometry = kwargs["geometry"]
    def fail(*args, **kwargs):
        raise AssertionError("Read-only prepared topology must not be reassembled")
    monkeypatch.setattr(np, "column_stack", fail)
    scaled = replace(geometry, attachment_vectors=geometry.attachment_vectors * 1.1,
                     landmark_vectors=geometry.landmark_vectors * 1.1)
    solver.positions(**dict(kwargs, geometry=scaled))
    with pytest.raises(ValueError, match="vectors must match"):
        solver.positions(**dict(kwargs, geometry=replace(scaled, landmark_vectors=scaled.landmark_vectors[:1])))
    with pytest.raises(ValueError, match="vectors must be finite"):
        solver.positions(**dict(kwargs, geometry=replace(scaled, attachment_vectors=scaled.attachment_vectors * np.nan)))


def test_replaced_readonly_topology_and_writable_mutations_are_revalidated():
    solver, kwargs = fixture()
    solver.positions(**kwargs)
    geometry = kwargs["geometry"]
    owners = geometry.landmark_owners.copy()
    writable = replace(geometry, landmark_owners=owners)
    solver.positions(**dict(kwargs, geometry=writable))
    assert solver._validated_topology is None
    owners[0] = (owners[0] + 1) % len(geometry.segment_names)
    with pytest.raises(ValueError, match="landmark axis and owners"):
        solver.positions(**dict(kwargs, geometry=writable))
    solver.positions(**kwargs)
    owners.setflags(write=False)
    with pytest.raises(ValueError, match="landmark axis and owners"):
        solver.positions(**dict(kwargs, geometry=replace(geometry, landmark_owners=owners)))


@pytest.mark.parametrize("mismatch", ["landmark_names", "landmark_owners", "landmark_index_type",
                                    "landmark_shape", "edge_shape", "edge_endpoint", "edge_identity",
                                    "edge_sign", "edge_index_type"])
def test_geometry_axes_and_edge_topology_must_match(mismatch):
    solver, kwargs = fixture()
    geometry = kwargs["geometry"]
    if mismatch == "landmark_names":
        geometry = replace(geometry, landmark_names=tuple(reversed(geometry.landmark_names)))
    elif mismatch == "landmark_owners":
        owners = geometry.landmark_owners.copy()
        owners[0] = (owners[0] + 1) % len(geometry.segment_names)
        geometry = replace(geometry, landmark_owners=owners)
    elif mismatch == "landmark_index_type":
        geometry = replace(geometry, landmark_owners=geometry.landmark_owners.astype(float))
    elif mismatch == "landmark_shape":
        geometry = replace(geometry, landmark_vectors=geometry.landmark_vectors[:1])
    elif mismatch == "edge_shape":
        geometry = replace(geometry, attachment_vectors=geometry.attachment_vectors[:1])
    else:
        levels = [[column.copy() for column in level] for level in geometry.traversal_levels]
        if mismatch == "edge_endpoint":
            levels[0][1][0] = levels[0][0][0]
        elif mismatch == "edge_identity":
            levels[0][2][0] = len(geometry.attachment_owners)
        elif mismatch == "edge_sign":
            levels[0][3][0] = 0
        else:
            levels[0][2] = levels[0][2].astype(float)
        geometry = replace(geometry, traversal_levels=tuple(tuple(level) for level in levels))
    with pytest.raises(ValueError, match="Geometry|Fitted geometry"):
        solver.positions(**dict(kwargs, geometry=geometry))


def test_cache_reuses_decompositions_and_is_bounded(monkeypatch):
    solver, kwargs = fixture(frames=300)
    rng = np.random.default_rng(82)
    present = rng.random(kwargs["segment_present"].shape) > .1
    present[:, solver.anchor_index] = True
    kwargs["segment_present"] = present
    kwargs["world_orientations"][:] = [1., 0., 0., 0.]
    kwargs["observed_origins"][:] = 0
    solver.positions(**kwargs)
    assert len(solver._operators) == 128
    single = {name: value[-1:] if isinstance(value, np.ndarray) else value for name, value in kwargs.items()}
    solver.positions(**single)
    def fail(*args, **kwargs):
        raise AssertionError("Cached pattern must not decompose again")
    monkeypatch.setattr(np.linalg, "svd", fail)
    solver.positions(**single)
