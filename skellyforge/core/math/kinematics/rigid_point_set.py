"""Multi-point rigid-body fit: MDS template construction + per-frame Procrustes.

A segment with ``landmarks`` of size ≥ 3 is a rigid body in the ordinary
mechanical sense: the pairwise distances between its points are invariant
under any motion.  This module is the numeric core of the *full* rigid-body
fit — build an invariant template once (from a distance matrix, either
measured or authored), then, per frame, find the best rigid placement of that
template onto the observed landmarks.

The template is built by Classical Multi-Dimensional Scaling (MDS):
double-center the squared distance matrix and take the leading eigenvectors.
MDS recovers positions from distances *up to* an arbitrary rotation /
reflection — a point set and its mirror share the same distance matrix.  The
sign (chirality) ambiguity is broken by Procrustes-aligning the raw MDS
embedding onto a *reference configuration* that lives in a known, hand-authored
orientation (the segment's reference geometry).  The reference configuration's
authored handedness is the source of truth — a left-handed reference would
silently mirror the model — so that guaranteed-correct chirality is what makes
``from_distances`` deterministically correct.

Adapted from the ferret-skull rigid-body solver in the bs client repo
(``clients/bs/python_code/rigid_body_solver/core/calculate_reference_geometry.py``),
specifically its ``estimate_distance_matrix`` + ``reconstruct_from_distances``.
The pyceres batch optimizer and its temporal-smoothness factors are
*deliberately not* ported: our realtime pipeline smooths keypoints upstream
(Euro filter) and orientations downstream (D3/D4 damping), so a per-frame
closed-form fit is the correct simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skellyforge.type_overloads import FloatArray

from skellyforge.kinematics.coordinate_frame_ops import align_point_sets_kabsch


def embed_distance_matrix(
    distance_matrix: FloatArray,
    n_dims: int = 3,
) -> FloatArray:
    """Recover point positions from a distance matrix via Classical MDS.

    Double-centers ``D²`` with ``H = I − 1/n``, eigen-decomposes, and scales
    the top ``n_dims`` eigenvectors by the square root of their (clamped ≥ 0)
    eigenvalues.  The returned embedding is centered at the origin; its
    orientation is arbitrary (MDS is invariant to rotation/reflection).

    Provenance: adapted from ``reconstruct_from_distances`` in the bs repo's
    ``calculate_reference_geometry.py`` (Classical MDS, straight port).

    Parameters
    ----------
    distance_matrix : (n, n)
        Symmetric matrix of pairwise distances (zeros on the diagonal).
    n_dims : int
        Embedding dimensionality (3 for spatial point sets).

    Returns
    -------
    (n, n_dims)
        Embedded positions, centered at the origin.
    """
    n = distance_matrix.shape[0]
    d_squared = distance_matrix**2
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d_squared @ h

    eigenvalues, eigenvectors = np.linalg.eigh(b)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = eigenvalues[:n_dims]
    eigenvectors = eigenvectors[:, :n_dims]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    coordinates = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    return coordinates


@dataclass(frozen=True)
class RigidPointTemplate:
    """The invariant geometry of a multi-point rigid body.

    A frozen snapshot of a rigid point set whose pairwise distances are the
    whole truth: ``positions`` is the embedded template (centered at the
    origin, in the reference configuration's chirality), and
    ``pair_distances`` is its canonical distance table keyed by sorted
    name-tuples (order-independent lookups).  ``positions`` and
    ``pair_distances`` are stored as defensive copies (and ``_index_by_name``
    is built once) so the frozen template is safe against in-place mutation
    by its callers and can be read per frame without rebuilding lookup tables.
    """

    point_names: tuple[str, ...]
    positions: FloatArray
    pair_distances: dict[tuple[str, str], float]

    # Built once in ``__post_init__`` so the per-frame fit does index lookups,
    # not dict construction.  ``frozen=True`` blocks rebound only — these
    # cached structures (and ``positions`` below) are defensive copies so
    # in-place mutation of the caller's arrays/dicts cannot corrupt the
    # invariant template.
    _index_by_name: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if len(self.point_names) != self.positions.shape[0]:
            raise ValueError(
                f"point_names ({len(self.point_names)}) and positions "
                f"({self.positions.shape[0]}) must have the same length"
            )
        if self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (n, 3), got {self.positions.shape}"
            )

        # Defensive copy: ``positions`` is a mutable ndarray on a frozen
        # dataclass, so copy it to preserve the template's invariance against
        # in-place mutation by the caller.
        object.__setattr__(self, "positions", np.array(self.positions, copy=True))
        # ``pair_distances`` values are plain floats (immutable), so a shallow
        # dict copy is enough to guard against exterior key mutation.
        object.__setattr__(self, "pair_distances", dict(self.pair_distances))
        object.__setattr__(
            self,
            "_index_by_name",
            {name: i for i, name in enumerate(self.point_names)},
        )

    @classmethod
    def from_distances(
        cls,
        point_names: list[str] | tuple[str, ...],
        pair_distances: dict[tuple[str, str], float],
        *,
        reference_configuration: dict[str, FloatArray] | None = None,
    ) -> RigidPointTemplate:
        """Build a template from pairwise distances.

        Assembles the (n, n) distance matrix — every pair present in
        ``pair_distances`` is used directly; a pair that is missing falls back
        to the distance implied by ``reference_configuration`` when it is
        provided, and otherwise raises (fail-loud: the caller must supply
        every rigid edge either way).

        The MDS embedding's sign/chirality is stabilized by
        Procrustes-aligning it onto ``reference_configuration`` (the segment's
        authored reference geometry, which lives in a known, hand-authored
        orientation).  When ``reference_configuration`` is ``None`` the raw
        embedding is returned with its arbitrary orientation — the caller's
        responsibility (a previous template's ``positions`` can be passed to
        keep rebuilds consistent across frames).

        Parameters
        ----------
        point_names :
            Ordered names of the rigid points (defines matrix row/column order).
        pair_distances :
            Known pairwise distances keyed by sorted name-tuples.  Any name
            here must be in ``point_names``.
        reference_configuration :
            A full reference placement ``{name: (3,) position}`` used to (a)
            fill missing pair distances and (b) resolve chirality.  Optional.

        Raises
        ------
        ValueError
            If a pair's distance cannot be resolved from ``pair_distances`` or
            ``reference_configuration`` (naming the offending pair), or if a
            pair references a name not in ``point_names``.
        """
        names = tuple(point_names)
        idx = {name: i for i, name in enumerate(names)}

        n = len(names)
        dm = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sorted((names[i], names[j]))
                key = (a, b)
                if key in pair_distances:
                    dm[i, j] = dm[j, i] = float(pair_distances[key])
                else:
                    ref_i = reference_configuration.get(a) if reference_configuration else None
                    ref_j = reference_configuration.get(b) if reference_configuration else None
                    if ref_i is not None and ref_j is not None:
                        dm[i, j] = dm[j, i] = float(np.linalg.norm(ref_i - ref_j))
                    else:
                        raise ValueError(
                            f"No distance for pair {key!r} — not in pair_distances "
                            f"and not recoverable from the reference configuration"
                        )

        # Sanity: any pair_distances key must reference known names.
        for key in pair_distances:
            if key[0] not in idx or key[1] not in idx:
                raise ValueError(
                    f"pair {key!r} references a name not in point_names: {names}"
                )

        embedded = embed_distance_matrix(dm, n_dims=3)

        if reference_configuration is not None:
            # Sign/chirality stabilization: rotate the embedding onto the
            # reference placement so the template adopts its known chirality.
            ref_positions = np.stack([reference_configuration[nm] for nm in names])
            # Kabsch returns the rotation taking the embedding onto the
            # reference (det = +1, so chirality is preserved, not mirrored).
            rotation = align_point_sets_kabsch(
                embedded, ref_positions.astype(np.float64)
            )
            embedded = (rotation @ embedded.T).T

        return cls(
            point_names=names,
            positions=embedded,
            pair_distances=dict(pair_distances),
        )

    @property
    def distance_matrix(self) -> FloatArray:
        """The (n, n) distance matrix implied by ``positions``."""
        n = len(self.point_names)
        dm = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dm[i, j] = dm[j, i] = d
        return dm


def fit_template_to_observed(
    template: RigidPointTemplate,
    observed: dict[str, FloatArray],
    *,
    anchor_name: str | None = None,
) -> dict[str, FloatArray]:
    """Place the template onto observed landmarks with a closed-form rigid fit.

    The common points are the template names present in ``observed`` with all
    finite values.  With fewer than 3 of them the rigid fit is
    under-determined and a copy of ``observed`` is returned unchanged
    (occlusion is data — the caller's fallback).

    With an ``anchor_name`` in the common set, the fit is rotation-only
    Procrustes pinned at the anchor: translate both point sets so the anchor
    is at the origin, Kabsch-align the template subset onto the observed
    subset, then ``corrected_i = anchor_observed + R @ (template_i − anchor_template)``
    — extrapolated to *all* template points (that is the point of a rigid
    body).  Without an ``anchor_name`` the fit is full R+t Procrustes
    (translation = centroid difference).

    Returns a dict keyed by *all* template point names.  A degenerate Kabsch
    (collinear / zero-variance common set) returns ``observed`` unchanged.

    Return contract: on the healthy path the dict is keyed by *template* point
    names (extrapolated to every template point); the fallback returns are
    keyed by *observed* names (a plain copy of ``observed``) because with an
    under-determined fit there is no template placement to extrapolate.

    Parameters
    ----------
    template :
        The invariant rigid geometry to place.
    observed :
        ``{name: (3,) position}`` for the observed landmarks.
    anchor_name :
        When given and among the common points, pin the corrected position at
        this anchor exactly and solve rotation only.

    Returns
    -------
    dict[str, (3,) FloatArray]
        Corrected positions for every template point name.
    """
    # Common points: present, finite.  Position lookups use the template's
    # cached ``_index_by_name`` (built once at load) rather than rebuilding a
    # name→index dict every frame.
    positions = template.positions
    index_by_name = template._index_by_name  # noqa: SLF001 — hot path
    common = [
        name
        for name in template.point_names
        if name in observed and np.all(np.isfinite(observed[name]))
    ]

    if len(common) < 3:
        return {name: observed[name] for name in observed}

    common_template = np.stack(
        [positions[index_by_name[name]] for name in common]
    ).astype(np.float64)
    common_observed = np.stack([observed[name] for name in common]).astype(np.float64)

    try:
        if anchor_name is not None and anchor_name in common:
            anchor_t = positions[index_by_name[anchor_name]]
            anchor_o = observed[anchor_name]
            t_centered = common_template - anchor_t
            o_centered = common_observed - anchor_o
            rotation = align_point_sets_kabsch(
                t_centered, o_centered
            )
            corrected = {
                name: anchor_o + rotation @ (positions[index_by_name[name]] - anchor_t)
                for name in template.point_names
            }
        else:
            t_centroid = common_template.mean(axis=0)
            o_centroid = common_observed.mean(axis=0)
            t_centered = common_template - t_centroid
            o_centered = common_observed - o_centroid
            rotation = align_point_sets_kabsch(
                t_centered, o_centered
            )
            corrected = {
                name: o_centroid
                + rotation @ (positions[index_by_name[name]] - t_centroid)
                for name in template.point_names
            }
    except ValueError:
        # Degenerate Kabsch (e.g. collinear points) — under-determined fit.
        return {name: observed[name] for name in observed}

    return corrected
