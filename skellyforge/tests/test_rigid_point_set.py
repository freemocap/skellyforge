"""Tests for the multi-point rigid-body fit (MDS template + Procrustes)."""

import numpy as np

from skellyforge.kinematics.coordinate_frame_ops import align_point_sets_kabsch
from skellyforge.kinematics.rigid_point_set import (
    RigidPointTemplate,
    embed_distance_matrix,
    fit_template_to_observed,
)


def _random_6_points(rng: np.random.Generator):
    """Six non-degenerate 3D points from a fixed-seed generator."""
    return rng.normal(size=(6, 3))


def _pair_distances(positions: np.ndarray, names: list[str]):
    d: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d[(names[i], names[j])] = float(np.linalg.norm(positions[i] - positions[j]))
    return d


# ── 1. Classical MDS ───────────────────────────────────────────────


def test_mds_round_trip_preserves_distances():
    rng = np.random.default_rng(0)
    points = _random_6_points(rng)
    names = [f"p{i}" for i in range(6)]
    distances = _pair_distances(points, names)
    dm = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            a, b = sorted((names[i], names[j]))
            dm[i, j] = distances[(a, b)]

    embedded = embed_distance_matrix(dm, n_dims=3)

    # embedding is centered
    assert np.allclose(embedded.mean(axis=0), 0.0, atol=1e-12)

    # pairwise distances preserved (up to the global MDS orientation)
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = sorted((names[i], names[j]))
            assert np.isclose(
                np.linalg.norm(embedded[i] - embedded[j]),
                distances[(a, b)],
                atol=1e-9,
            )


# ── 2. Template chirality / sign stabilization ────────────────────


def test_template_reference_chirality():
    # A set and its mirror share the same distance matrix.  The reference
    # configuration (unmirrored) must win: the template must match the
    # unmirrored chirality, not the mirror.
    rng = np.random.default_rng(0)
    points = _random_6_points(rng)
    mirror = points.copy()
    mirror[:, 2] *= -1.0
    names = ["a", "b", "c", "d", "e", "f"]

    distances = _pair_distances(points, names)  # identical for mirror
    reference = {n: points[i] for i, n in enumerate(names)}

    template = RigidPointTemplate.from_distances(
        names, distances, reference_configuration=reference
    )

    # The template is centered at the origin; the reference is NOT.  Center
    # both before measuring the rotation-only Kabsch residual.  Aligning
    # template -> centered-reference should give ~0 residual; aligning to the
    # centered MIRROR is the wrong chirality — a det+1 rotation cannot map the
    # template onto its mirror, so the residual stays large.  Normalize the
    # two residuals into a scale-invariant ratio: the reference must be ~100×
    # closer than the mirror, independent of geometry magnitude or execution
    # order.
    ref_centered = points - points.mean(axis=0)
    mirror_centered = mirror - mirror.mean(axis=0)
    R = align_point_sets_kabsch(template.positions, ref_centered)
    ref_residual = np.linalg.norm((R @ template.positions.T).T - ref_centered)
    R_m = align_point_sets_kabsch(template.positions, mirror_centered)
    mirror_residual = np.linalg.norm((R_m @ template.positions.T).T - mirror_centered)

    assert ref_residual < 1e-9
    assert ref_residual / mirror_residual < 0.01


# ── 3/4. Per-frame fit preserves distances + is least-squares ─────


def _template_and_observed(rng_key=1):
    rng = np.random.default_rng(rng_key)
    names = ["a", "b", "c", "d", "e"]
    points = rng.normal(size=(5, 3))
    distances = _pair_distances(points, names)
    reference = {n: points[i] for i, n in enumerate(names)}
    template = RigidPointTemplate.from_distances(
        names, distances, reference_configuration=reference
    )
    # known R (proper rotation) + translation t + small noise
    R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = np.array([1.0, 2.0, 3.0])
    noise = rng.normal(scale=1e-3, size=(5, 3))
    observed_positions = (R @ points.T).T + t + noise
    observed = {n: observed_positions[i] for i, n in enumerate(names)}
    return template, observed, points, noise, R, t


def test_fit_preserves_all_pairwise_distances_exactly():
    template, observed, *_ = _template_and_observed()
    corrected = fit_template_to_observed(template, observed)
    assert set(corrected) == set(template.point_names)
    # pairwise distances in corrected must equal template distances exactly
    for i in range(len(template.point_names)):
        for j in range(i + 1, len(template.point_names)):
            n_i = template.point_names[i]
            n_j = template.point_names[j]
            expected = template.pair_distances[tuple(sorted((n_i, n_j)))]
            assert np.isclose(
                np.linalg.norm(corrected[n_i] - corrected[n_j]),
                expected,
                atol=1e-9,
            )


def test_fit_is_least_squares_closest():
    template, observed, _, noise, *_ = _template_and_observed()
    corrected = fit_template_to_observed(template, observed)
    # The fit is (closed-form) least-squares: its squared residual vs the
    # noisy observations must be <= the squared residual of the TRUE rigid
    # placement (R @ points + t) vs the same noisy observations — i.e. the fit
    # cannot be worse than the ground truth embedded in the noise.
    fit_residual = sum(
        np.dot(corrected[n] - observed[n], corrected[n] - observed[n])
        for n in template.point_names
    )
    truth_residual = sum(
        np.dot(noise[i], noise[i]) for i in range(noise.shape[0])
    )
    assert fit_residual <= truth_residual + 1e-12


# ── 5. Anchor pinning ─────────────────────────────────────────────


def test_anchor_pins_the_anchor():
    template, observed, *_ = _template_and_observed()
    anchor = "b"
    corrected = fit_template_to_observed(template, observed, anchor_name=anchor)
    assert np.allclose(corrected[anchor], observed[anchor], atol=1e-12)


# ── 6. Extrapolation of missing points ────────────────────────────


def test_fit_extrapolates_missing_points():
    template, observed, *_ = _template_and_observed()
    # drop one point from observed — it must still get a corrected position
    del observed["d"]
    corrected = fit_template_to_observed(template, observed, anchor_name="a")
    assert "d" in corrected
    assert np.all(np.isfinite(corrected["d"]))
    # and the visible points' pairwise distances still hold
    for n_i, n_j in [("a", "b"), ("a", "c"), ("b", "c"), ("c", "e")]:
        expected = template.pair_distances[tuple(sorted((n_i, n_j)))]
        assert np.isclose(
            np.linalg.norm(corrected[n_i] - corrected[n_j]),
            expected,
            atol=1e-9,
        )


# ── 7. Under-determined passthrough ───────────────────────────────


def test_fewer_than_three_common_points_passthrough():
    template, observed, *_ = _template_and_observed()
    observed_subset = {"a": observed["a"], "b": observed["b"]}
    result = fit_template_to_observed(template, observed_subset)
    assert set(result) == set(observed_subset)
    for n in observed_subset:
        assert np.array_equal(result[n], observed_subset[n])


# ── 8. Unanchored fit translates ──────────────────────────────────


def test_unanchored_fit_translates():
    template, observed, *_ = _template_and_observed()
    corrected = fit_template_to_observed(template, observed)
    corrected_centroid = np.mean(
        np.stack([corrected[n] for n in template.point_names]), axis=0
    )
    observed_centroid = np.mean(
        np.stack([observed[n] for n in template.point_names]), axis=0
    )
    assert np.allclose(corrected_centroid, observed_centroid, atol=1e-9)


# ── 9. No reflection ──────────────────────────────────────────────


def test_no_reflection():
    template, observed, points, noise, R, t = _template_and_observed(rng_key=2)
    # clean case (no noise) so R reconstruction is exact
    observed_clean = {n: (R @ points[i]).T + t for i, n in enumerate(template.point_names)}
    anchor = "a"
    corrected = fit_template_to_observed(template, observed_clean, anchor_name=anchor)

    anchor_t = template.positions[template.point_names.index(anchor)]
    anchor_c = corrected[anchor]

    T = np.stack([template.positions[i] for i in range(len(template.point_names))]) - anchor_t
    C = np.stack([corrected[n] for n in template.point_names]) - anchor_c

    R_fit = _procrustes_rotation(T, C)
    assert np.linalg.det(R_fit) > 0.0


def _procrustes_rotation(src, dst):
    """Minimal R (no reflection guard) for det-check in tests."""
    h = src.T @ dst
    U, _, Vt = np.linalg.svd(h)
    return Vt.T @ U.T
