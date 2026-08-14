"""Runtime coordinate-frame construction from live keypoint positions.

Where ``reference_geometry.SegmentReferenceGeometry`` stores the **static**
T-pose frame (exact axis + approximate axis), this module provides the
**dynamic** operations that build frames from live data and compute the
rotations that relate them to the reference.

Used by the orientation solver (SF-SH-4) to produce per-segment quaternions
from the live skeleton vs the ``SegmentReferenceGeometry`` in the standard
human model.

Operations
----------
- ``build_segment_frame`` — construct a segment's frame from its tagged axis
  declarations + positions (the ONE builder; dispatch on 1/2/3 axes).
- ``rotation_between_vectors`` — shortest rotation that aligns one unit
  vector onto another (swing-only; no twist).
- ``align_point_sets_kabsch`` — Kabsch (Umeyama) alignment of two
  corresponding point clouds → optimal rotation matrix.
- ``compute_rotation_from_live_basis`` — rotation quaternion from a live
  basis to a reference basis.

References
----------
- Kabsch (1976) "A solution for the best rotation to relate two sets of
  vectors" — point-set alignment via SVD.
- Umeyama (1991) "Least-squares estimation of transformation parameters
  between two point patterns" — reflection correction for Kabsch.
- Gram-Schmidt orthogonalization, standard linear algebra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.quaternion_math import (
    RotationQuaternion,
    normalize_quaternion_array,
)

if TYPE_CHECKING:
    from numpy import float64
    from skellyforge.skellymodels.standard_human.segment_definition import (
        AxisDefinition,
        AxisKind,
    )


# ── Basis construction ───────────────────────────────────────────────


def build_segment_frame(
    axes: tuple["AxisDefinition", ...],
    positions: dict[str, NDArray[float64]],
    origin_keypoint: str,
    *,
    collinearity_threshold: float = 0.9998,
) -> tuple[NDArray[float64] | None, bool]:
    """Build a segment's local frame from its tagged axis declarations + positions.

    The ONE builder used to construct a segment frame from declared axes. The
    axis NAME (x/y/z) selects which basis vector a declaration defines; the
    KIND (EXACT/APPROXIMATE) selects how its direction feeds the construction.
    The exact axis may be declared on any of x/y/z — there is no positional
    assumption.

    Every axis direction is ``positions[target_keypoint] −
    positions[origin_keypoint]`` — the segment's origin to a point of its own
    rigid geometry, normalized.

    Construction builds in TWO PASSES, each in basis order (x, y, z), not the
    authored tuple order:

    - Pass 1 places every EXACT axis's direction as the hard value of its named
      basis vector (``x̂``/``ŷ``/``ẑ``).
    - Pass 2 Gram-Schmidt-projects every APPROXIMATE axis's direction against
      ALL vectors built so far, filling its named vector with the residual.
    - The third (undeclared) vector fills via the right-handed cross product of
      the other two; a declaration on that name resolves only its sign
      (dot-product consistency: if the dot is negative, flip it) — never its
      direction.
    - One declared axis yields an incomplete frame (``resolved`` is ``False``).

    A single declared axis (one exact) yields an incomplete frame (``resolved``
    is ``False``) — the roll is not determined. An APPROXIMATE direction that is
    collinear with the exact direction degrades softly (``resolved`` is
    ``False``); an EXACT direction that is collinear with another already-built
    vector RAISES — an exact declaration that cannot distinguish anything is a
    wrong declaration.

    Parameters
    ----------
    axes :
        The segment's tagged axis declarations (1–3 of them, already validated).
    positions :
        ``{keypoint_name: (3,) position}`` — rest positions for reference
        geometry, live positions for the solver. A keypoint missing from this
        map makes its axis unusable (see ``resolved``/``raises`` below).
    origin_keypoint :
        The segment's origin keypoint — the start of every axis direction.
    collinearity_threshold :
        Dot-product bound beyond which an APPROXIMATE direction is treated as
        collinear (hence unresolved → ``(None, False)``) rather than a hard
        error. The solver passes the ~5° singularity-gate threshold (``0.996``);
        the reference geometry passes the stiffer ~1° ``0.9998``. EXACT-axis
        collinearity ALWAYS raises regardless of this threshold.

    Returns
    -------
    (basis, resolved)
        ``basis`` is a right-handed ``(3, 3)`` frame with rows
        [x̂, ŷ, ẑ] when ``resolved`` is ``True``; ``None`` otherwise.
        ``resolved`` is ``False`` when the frame is incomplete (a single exact
        axis) or an approximate direction is unusable/collinear this build.
    """
    if not axes:
        raise ValueError("build_segment_frame needs at least one axis")

    # Deferred import: segment_definition pulls in the standard_human package,
    # whose __init__ reverse-imports reference_geometry → this module, so a
    # top-level import here would cycle.
    from skellyforge.skellymodels.standard_human.segment_definition import (
        AxisKind,
    )

    # One declared axis → incomplete frame (roll unresolved).
    if len(axes) == 1:
        return None, False

    # Index declarations by name (validated distinct); each maps to one basis
    # vector.
    by_name: dict[str, "AxisDefinition"] = {a.axis: a for a in axes}

    origin = positions.get(origin_keypoint)
    if origin is None:
        return None, False
    origin = np.asarray(origin, dtype=np.float64)

    def _direction(target: str) -> NDArray[float64] | None:
        """origin → target, normalized; ``None`` if absent or degenerate."""
        to = positions.get(target)
        if to is None:
            return None
        vec = np.asarray(to, dtype=np.float64) - origin
        norm = float(np.linalg.norm(vec))
        if norm < 1e-10:
            return None
        return vec / norm

    # Resolve every declared direction first (so names — not order — decide
    # which hard/soft directions land on which basis vector).
    dirs: dict[str, NDArray[float64]] = {}
    for name, decl in by_name.items():
        d = _direction(decl.target_keypoint)
        if d is None:
            return None, False
        dirs[name] = d

    # ── Assemble in TWO PASSES (x, y, z name order within each) ─────────
    # Pass 1 builds every declared EXACT axis as a HARD vector on its named basis
    # axis; pass 2 then Gram-Schmidt-projects every declared APPROXIMATE axis
    # against ALL vectors built so far. This ordering is load-bearing: an exact
    # axis is the segment's defining direction regardless of which basis name it
    # lands on, so it must always be hard — even when an approximate axis is
    # declared on an earlier basis name (approximate-x + exact-y, the hips and
    # toes). Building approximate-first would let a soft direction become the
    # hard first vector and the exact would never orthogonalize against it.
    built: dict[str, NDArray[float64]] = {}
    sign_hint: tuple[str, NDArray[float64]] | None = None

    def _project(vec: NDArray[float64], against: list[NDArray[float64]]) -> NDArray[float64] | None:
        """Gram-Schmidt-project vec against already-built vectors; None if it
        collapses below a numerical floor."""
        for v in against:
            vec = vec - float(np.dot(vec, v)) * v
        norm = float(np.linalg.norm(vec))
        if norm < 1e-10:
            return None
        return vec / norm

    # Pass 1: every EXACT direction is hard; an exact direction that collides
    # (collinear) with another already-built exact vector is a wrong declaration.
    for name in ("x", "y", "z"):
        decl = by_name.get(name)
        if decl is None or decl.kind is not AxisKind.EXACT:
            continue
        d = dirs[name]
        for v in built.values():
            if abs(float(np.dot(d, v))) > collinearity_threshold:
                raise ValueError(
                    "build_segment_frame: an EXACT axis is collinear with "
                    f"a previously built axis (|dot| = {abs(float(np.dot(d, v))):.6f}). "
                    "An exact declaration that cannot distinguish anything "
                    "is a wrong declaration."
                )
        built[name] = d

    # Pass 2: every APPROXIMATE direction is a soft reference — it projects
    # against every vector built so far (its named vector is the residual). A
    # singular (collinear) or below-floor residual degrades softly (unresolved),
    # not raise.
    for name in ("x", "y", "z"):
        decl = by_name.get(name)
        if decl is None or decl.kind is not AxisKind.APPROXIMATE:
            continue
        d = dirs[name]
        if len(built) == 2:
            # A third declaration (two named vectors already built) is a
            # sign-only hint for the cross-produced vector — its direction does
            # NOT build a third hard vector.
            sign_hint = (name, d)
            continue
        if built:
            for v in built.values():
                if abs(float(np.dot(d, v))) > collinearity_threshold:
                    return None, False  # singularity gate — soft degrade
            proj = _project(d, list(built.values()))
            if proj is None:
                return None, False
            built[name] = proj
        else:
            built[name] = d

    # Complete the frame: fewer than two named vectors → roll unresolved.
    if len(built) < 2:
        return None, False

    named_idx = {_AXIS_TO_INDEX[n]: v for n, v in built.items()}
    basis = assemble_named_basis(named_idx)

    # A third declaration resolves only the sign of the cross-produced vector.
    if sign_hint is not None and len(built) == 2:
        hint_name, hint_dir = sign_hint
        idx = _AXIS_TO_INDEX[hint_name]
        if float(np.dot(hint_dir, basis[idx])) < 0.0:
            basis = basis.copy()
            basis[idx] = -basis[idx]

    return basis, True


# ── Named-row basis assembly (shared by reference geometry + solver) ──

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def assemble_named_basis(named: dict[int, NDArray[float64]]) -> NDArray[float64]:
    """Build a right-handed (3,3) basis from two named unit rows + one cross.

    Two of the three rows (indexed [x̂, ŷ, ẑ] = 0,1,2) are supplied as already
    orthonormalized unit vectors; the missing row fills via the right-handed
    cross product of the other two in cyclic order. This is the shared
    assembler both the reference geometry and the solver use so their frames
    agree row-for-row at the T-pose.
    """
    missing = [i for i in (0, 1, 2) if i not in named]
    if len(missing) != 1:
        raise ValueError("assemble_named_basis needs exactly two named rows")
    m = missing[0]
    nxt = (m + 1) % 3
    prev = (m + 2) % 3
    # v_m = v_nxt × v_prev (right-handed cyclic: ŷ×ẑ=x̂, ẑ×x̂=ŷ, x̂×ŷ=ẑ)
    vec_m = np.cross(named[nxt], named[prev])
    norm = float(np.linalg.norm(vec_m))
    if norm < 1e-10:
        raise ValueError("named basis rows are collinear")
    vec_m = vec_m / norm
    out = np.empty((3, 3), dtype=np.float64)
    for i in (0, 1, 2):
        out[i] = named[i] if i in named else vec_m
    return out


# ── Swing rotation ───────────────────────────────────────────────────


def rotation_between_vectors(
    from_vector: NDArray[float64],
    to_vector: NDArray[float64],
) -> RotationQuaternion:
    """Compute the shortest rotation that aligns *from_vector* onto *to_vector*.

    This is a **swing-only** rotation — it aligns the long axes of two
    configurations but leaves the twist (rotation around the axis)
    undetermined. The twist is resolved separately by the per-bone
    declaration: a declared APPROXIMATE direction reference (the first tier),
    or the damped minimal-roll fallback (the second tier).

    Parameters
    ----------
    from_vector : (3,)
        Unit vector in the reference configuration (e.g. T-pose bone direction).
    to_vector : (3,)
        Unit vector in the live configuration.

    Returns
    -------
    RotationQuaternion
        The rotation taking *from_vector* to *to_vector*.

    Notes
    -----
    If the vectors are anti-parallel (dot < −0.9999), any axis
    perpendicular to *from_vector* is a valid rotation axis — we pick
    one via Gram-Schmidt against an arbitrary non-collinear basis vector.
    """
    a = np.asarray(from_vector, dtype=np.float64)
    b = np.asarray(to_vector, dtype=np.float64)
    _check_unit_vector(a, "from_vector")
    _check_unit_vector(b, "to_vector")

    dot = float(np.dot(a, b))
    cross = np.cross(a, b)
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-10:
        # Vectors are parallel (dot ≈ 1) or anti-parallel (dot ≈ −1)
        if dot > 0.0:
            return RotationQuaternion.identity()

        # Anti-parallel: pick any perpendicular axis via Gram-Schmidt
        # against an arbitrary non-collinear basis vector
        arbitrary = (
            np.array([0.0, 1.0, 0.0])
            if abs(a[0]) > 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        axis = arbitrary - np.dot(arbitrary, a) * a
        axis = axis / np.linalg.norm(axis)
        # 180° rotation around that axis: quaternion (0, axis)
        return RotationQuaternion(w=0.0, x=float(axis[0]), y=float(axis[1]), z=float(axis[2]))

    # Standard case: axis = cross(a, b), angle = arccos(dot)
    axis = cross / cross_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    return RotationQuaternion(
        w=float(np.cos(half_angle)),
        x=float(axis[0] * sin_half),
        y=float(axis[1] * sin_half),
        z=float(axis[2] * sin_half),
    )


# ── Kabsch alignment (full-frame bones) ──────────────────────────────


def align_point_sets_kabsch(
    reference_points: NDArray[float64],
    live_points: NDArray[float64],
) -> NDArray[float64]:
    """Find the optimal rotation aligning two corresponding point sets.

    Kabsch algorithm with Umeyama's reflection correction: computes the
    rotation matrix R that minimizes ``||R @ P_ref - P_live||²`` for
    centered point sets.

    Parameters
    ----------
    reference_points : (M, 3)
        Point positions in the reference (T-pose) configuration.
    live_points : (M, 3)
        Corresponding point positions in the live configuration.

    Returns
    -------
    (3, 3) float64
        Optimal rotation matrix (right-handed, det = +1).

    Raises
    ------
    ValueError
        If fewer than 3 points are provided or if the points are
        collinear / degenerate.

    References
    ----------
    - Kabsch (1976) "A solution for the best rotation to relate two sets
      of vectors", Acta Crystallographica A32:922–923.
    - Umeyama (1991) "Least-squares estimation of transformation
      parameters between two point patterns", IEEE PAMI 13(4):376–380.
    """
    P = np.asarray(reference_points, dtype=np.float64)
    Q = np.asarray(live_points, dtype=np.float64)

    if P.shape != Q.shape:
        raise ValueError(
            f"Point sets must have the same shape, "
            f"got reference {P.shape} vs live {Q.shape}"
        )
    if P.shape[0] < 3:
        raise ValueError(
            f"Kabsch requires ≥3 points for unique rotation, got {P.shape[0]}"
        )
    if P.shape[1] != 3:
        raise ValueError(
            f"Points must have shape (M, 3), got {P.shape}"
        )

    # Center both point sets
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Cross-covariance matrix H = P^T @ Q
    H = P_centered.T @ Q_centered  # (3, 3)

    # SVD: H = U @ S @ V^T
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Optimal rotation (before reflection check)
    R = V @ U.T

    # Umeyama reflection correction: if det(R) < 0, flip the last
    # column of V and recompute.
    if np.linalg.det(R) < 0.0:
        V_corrected = V.copy()
        V_corrected[:, -1] *= -1.0
        R = V_corrected @ U.T

    # Sanity check
    if not np.allclose(np.linalg.det(R), 1.0, atol=1e-10):
        raise RuntimeError(
            f"Kabsch produced rotation with det = {np.linalg.det(R):.6e} "
            f"(expected +1). Input points may be degenerate."
        )

    return R


# ── Live basis rotation ──────────────────────────────────────────────


def compute_rotation_from_live_basis(
    live_basis: NDArray[float64],
    reference_basis: NDArray[float64],
) -> RotationQuaternion:
    """Compute the rotation quaternion from a live basis to a reference basis.

    Given two right-handed orthonormal bases (rows = basis vectors), find
    the rotation R such that ``R @ reference_basis[i] ≈ live_basis[i]``
    for each axis i.

    This is the rotation that takes a bone from its T-pose reference
    orientation to its current live orientation.

    Parameters
    ----------
    live_basis : (3, 3)
        Orthonormal basis in the live configuration.
    reference_basis : (3, 3)
        Orthonormal basis in the reference (T-pose) configuration.

    Returns
    -------
    RotationQuaternion
        The rotation from reference to live configuration. Identity
        means the bone is exactly in its T-pose orientation.
    """
    live = np.asarray(live_basis, dtype=np.float64)
    ref = np.asarray(reference_basis, dtype=np.float64)

    if live.shape != (3, 3):
        raise ValueError(
            f"live_basis must be (3, 3), got {live.shape}"
        )
    if ref.shape != (3, 3):
        raise ValueError(
            f"reference_basis must be (3, 3), got {ref.shape}"
        )

    # R = live_basis^T @ reference_basis? Or R = live_basis @ reference_basis^T?
    #
    # The reference basis rows are the axis directions at T-pose.
    # The live basis rows are the same axes in the current configuration.
    #
    # We want R such that for a vector v in reference coordinates:
    #   v_live = R @ v_ref
    #
    # The reference basis matrix (rows = axes) maps from axis-index
    # coordinates to Cartesian: axis_vec_ref = ref_basis @ e_i = ref_basis[i]
    # Actually the rows ARE the axis vectors in world space.
    #
    # We want R @ ref_basis[i] = live_basis[i] for each row i.
    # In matrix form: R @ ref_basis^T = live_basis^T (treating rows as column vectors)
    # So: R = live_basis^T @ inv(ref_basis^T)
    # Since ref_basis is orthonormal, ref_basis^T = ref_basis^{-1}
    # Wait, let me be more careful.
    #
    # Each row of the basis matrix is a unit vector in world space.
    # Row 0 = exact axis direction in world coords.
    #
    # We want R such that applying R to the reference axis gives the live axis:
    #   R @ ref[0] = live[0]
    #   R @ ref[1] = live[1]
    #   R @ ref[2] = live[2]
    #
    # Packing: let B_ref be (3,3) with rows = axis directions.
    # We want R @ B_ref^T = B_live^T  (treat rows as columns for matrix multiply)
    # So R = B_live^T @ (B_ref^T)^{-1} = B_live^T @ B_ref
    #
    # Since B_ref is orthonormal (rows are mutually orthogonal unit vectors),
    # B_ref^{-1} = B_ref^T, i.e., B_ref @ B_ref^T = I.
    # So (B_ref^T)^{-1} = B_ref.
    #
    # Therefore: R = B_live^T @ B_ref
    R = live.T @ ref

    # Numerical cleanup: ensure R is a proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0.0:
        Vt_corrected = Vt.copy()
        Vt_corrected[-1, :] *= -1.0
        R_clean = U @ Vt_corrected

    return RotationQuaternion.from_rotation_matrix(R_clean)


# ── Helpers ──────────────────────────────────────────────────────────


def _check_unit_vector(vec: NDArray[float64], name: str) -> None:
    """Raise ValueError if *vec* is not a unit (3,) vector."""
    if vec.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,), got {vec.shape}"
        )
    norm = float(np.linalg.norm(vec))
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(
            f"{name} must be a unit vector, got norm = {norm:.6f}"
        )
