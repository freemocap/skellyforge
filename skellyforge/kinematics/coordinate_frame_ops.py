"""Runtime coordinate-frame construction from live landmark positions.

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
- ``build_orthonormal_basis`` — construct a right-handed frame from two
  direction vectors (Gram-Schmidt + cross product).
- ``rotation_between_vectors`` — shortest rotation that aligns one unit
  vector onto another (swing-only; no twist).
- ``align_point_sets_kabsch`` — Kabsch (Umeyama) alignment of two
  corresponding point clouds → optimal rotation matrix.
- ``compute_live_basis_from_landmarks`` — build a live-configuration
  coordinate frame from tracked landmark positions against a reference
  geometry definition.
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

    The ONE builder used to construct a segment frame from declared axes — the
    tags (``AxisKind``), not the axis names, carry the roles.

    Every axis direction is ``positions[target_keypoint] −
    positions[origin_keypoint]`` — the segment's origin to a point of its own
    rigid geometry, normalized.

    Dispatch table (axes in authored order):

    ========  =================================================================
    Count     Behaviour
    ========  =================================================================
    1 axis    The sole (exact) axis gives ``x̂``. INCOMPLETE frame — roll is not
    (exact)   resolved (``resolved=False``); the damped minimal-roll tier takes
              over.
    2 axes    ``x̂`` = the first exact axis's direction. ``ŷ`` = Gram-Schmidt
    (exact +  projection of the second (approximate) direction against ``x̂``;
    approx)   ``ẑ`` = ``x̂ × ŷ``. Mirrors ``build_orthonormal_basis``.
    2 axes    Same projection math as exact + approximate, BUT if the second
    (exact +  exact direction is collinear with ``x̂`` at build time (rest or
    exact)    live), RAISE — an exact declaration that cannot distinguish
              anything is a wrong declaration.
    3 axes    ``x̂``, ``ŷ`` as the 2-axis case; ``ẑ`` = ``x̂ × ŷ``. The third
              declaration's direction is used ONLY to resolve ``ẑ``'s sign
              (dot-product consistency: if the dot is negative, flip ``ẑ``).
    ========  =================================================================

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
        ``resolved`` is ``False`` when the frame is incomplete (1 exact axis) or
        the approximate direction is unusable/collinear this build; the caller
        (the solver's damped-minimal tier) supplies the roll then.
    """
    if not axes:
        raise ValueError("build_segment_frame needs at least one axis")

    # 1 axis → incomplete frame (roll unresolved)
    if len(axes) == 1:
        return None, False

    exact = [a for a in axes if a.kind.value == "exact"]
    if not exact:
        raise ValueError("build_segment_frame: no EXACT axis among the declarations")

    origin = positions.get(origin_keypoint)
    if origin is None:
        return None, False
    origin = np.asarray(origin, dtype=np.float64)

    x_axis_def = exact[0]
    x_to = positions.get(x_axis_def.target_keypoint)
    if x_to is None:
        return None, False
    x_hat = np.asarray(x_to, dtype=np.float64) - origin
    x_norm = float(np.linalg.norm(x_hat))
    if x_norm < 1e-10:
        return None, False
    x_hat = x_hat / x_norm

    second = axes[1]
    s_to = positions.get(second.target_keypoint)
    if s_to is None:
        return None, False
    s_dir = np.asarray(s_to, dtype=np.float64) - origin
    s_norm = float(np.linalg.norm(s_dir))
    if s_norm < 1e-10:
        return None, False
    s_dir = s_dir / s_norm

    dot = float(np.dot(x_hat, s_dir))
    if abs(dot) > collinearity_threshold:
        if second.kind.value == "exact":
            # 2 exact axes, collinear at build time → fail loud
            raise ValueError(
                "build_segment_frame: the second EXACT axis is collinear with "
                f"the first exact axis (|dot| = {abs(dot):.6f}). An exact "
                "declaration that cannot distinguish anything is a wrong "
                "declaration."
            )
        # APPROXIMATE direction collinear → soft degradation (the singularity
        # gate): the roll does not resolve this build.
        return None, False

    basis = build_orthonormal_basis(x_hat, s_dir)

    # 3 axes → use the third declaration's direction only to resolve ẑ's sign
    if len(axes) >= 3:
        third = axes[2]
        t_to = positions.get(third.target_keypoint)
        if t_to is not None:
            t_dir = np.asarray(t_to, dtype=np.float64) - origin
            t_norm = float(np.linalg.norm(t_dir))
            if t_norm > 1e-10:
                t_dir = t_dir / t_norm
                if float(np.dot(t_dir, basis[2])) < 0.0:
                    basis = basis.copy()
                    basis[2] = -basis[2]

    return basis, True


def build_orthonormal_basis(
    exact_direction: NDArray[float64],
    approximate_direction: NDArray[float64],
) -> NDArray[float64]:
    """Build a right-handed orthonormal (3, 3) basis from two direction vectors.

    The **exact** direction becomes the first basis row unchanged.
    The **approximate** direction is Gram-Schmidt orthogonalized against
    the exact direction to remove the parallel component. The third axis
    is ``cross(exact, approximate)``, completing the right-handed frame.

    Parameters
    ----------
    exact_direction : (3,)
        Unit vector for the first basis axis. Must be pre-normalized.
    approximate_direction : (3,)
        Unit vector for the second axis direction. Need not be perfectly
        orthogonal to *exact_direction* — the parallel component is
        removed. Must not be parallel to *exact_direction*.

    Returns
    -------
    (3, 3) float64
        Rows are [exact_axis, orthogonalized_approximate, third_axis].

    Raises
    ------
    ValueError
        If the two directions are parallel (within ~1°).
    """
    exact = np.asarray(exact_direction, dtype=np.float64)
    approx = np.asarray(approximate_direction, dtype=np.float64)
    _check_unit_vector(exact, "exact_direction")
    _check_unit_vector(approx, "approximate_direction")

    # Gram-Schmidt: remove parallel component from approximate
    dot = float(np.dot(exact, approx))
    if abs(dot) > 0.9998:  # cos(1°) ≈ 0.9998
        raise ValueError(
            f"Exact and approximate directions are nearly parallel "
            f"(|dot| = {abs(dot):.6f}). Choose a different approximate "
            f"direction."
        )
    approx_orth = approx - dot * exact
    approx_orth = approx_orth / np.linalg.norm(approx_orth)

    # Third axis via right-handed cross product
    third = np.cross(exact, approx_orth)
    # Cross product of orthonormal vectors should be unit, but re-normalize
    # for numerical safety
    third_norm = float(np.linalg.norm(third))
    if third_norm < 1e-10:
        raise ValueError(
            "Cross product produced near-zero vector — axes may be "
            "parallel or one may be zero-length."
        )
    third = third / third_norm

    basis = np.empty((3, 3), dtype=np.float64)
    basis[0] = exact
    basis[1] = approx_orth
    basis[2] = third
    return basis


# ── Swing rotation ───────────────────────────────────────────────────


def rotation_between_vectors(
    from_vector: NDArray[float64],
    to_vector: NDArray[float64],
) -> RotationQuaternion:
    """Compute the shortest rotation that aligns *from_vector* onto *to_vector*.

    This is a **swing-only** rotation — it aligns the long axes of two
    configurations but leaves the twist (rotation around the axis)
    undetermined. The twist is resolved separately per the bone's
    ``TwistPolicy``.

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


# ── Live basis from landmarks ────────────────────────────────────────


def compute_live_bone_basis(
    live_bone_direction: NDArray[float64],
    live_twist_reference: NDArray[float64],
) -> NDArray[float64]:
    """Build a live-configuration coordinate basis for a bone.

    Takes the live bone's long axis (exact) and a twist-reference
    direction (approximate) and builds a right-handed orthonormal basis
    via ``build_orthonormal_basis``.

    Parameters
    ----------
    live_bone_direction : (3,)
        Unit vector along the bone's long axis (proximal → distal) in
        the live configuration.
    live_twist_reference : (3,)
        Unit vector for the twist-reference direction in the live
        configuration. What this IS depends on the bone's twist policy:
        - Full-frame: derived from ≥3 landmark positions on the segment.
        - Chain-resolved: the child bone's direction (e.g. forearm for
          upper arm, shank for thigh).
        - Damped-minimal: the reference approximate axis rotated by the
          swing rotation (and temporally damped).

    Returns
    -------
    (3, 3) float64
        Rows are [exact_axis, orthogonalized_approximate, third_axis].
    """
    return build_orthonormal_basis(live_bone_direction, live_twist_reference)


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
