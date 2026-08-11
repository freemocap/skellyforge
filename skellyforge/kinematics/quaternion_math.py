"""RotationQuaternion algebra for 3D rotations — scalar and vectorized operations.

This is the single home for all quaternion math in the kinematics engine.
Every operation lives here exactly once: scalar ``RotationQuaternion`` dataclass for
single-frame use, and pure numpy module-level functions for batch operations
on ``(N, 4)`` arrays. No other module in the project should contain a
Hamilton product, SLERP, or quaternion-to-matrix conversion.

Convention
----------
- Scalar-first ordering: **[w, x, y, z]** everywhere — on the wire, in
  numpy arrays, and in ``RotationQuaternion`` field order. This matches the
  standard stream's ``ROTATIONS_WORLD`` and ``ROTATIONS_LOCAL`` channel
  layout (``w, x, y, z`` float32 columns per doc 09).
- All quaternions are **unit** quaternions representing rotations. The
  scalar ``RotationQuaternion`` auto-normalizes in ``__post_init__``; vectorized
  functions assume pre-normalized input (call ``normalize_quaternion_array``
  if needed).
- Identity rotation is ``(1, 0, 0, 0)`` — this is the T-pose contract for
  every bone in the standard human model.

References
----------
- Hamilton (1844) "On Quaternions" — Hamilton product definition.
- Shoemake (1985) "Animating Rotation with RotationQuaternion Curves" — SLERP.
- Shepperd (1978) "RotationQuaternion from Rotation Matrix" — trace-based
  matrix-to-quaternion in ``from_rotation_matrix``.
- Diebel (2006) "Representing Attitude: Euler Angles, Unit Quaternions,
  and Rotation Vectors" — Euler ZYX intrinsic convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy import float64


# ═══════════════════════════════════════════════════════════════════════
# Scalar RotationQuaternion (slot-based dataclass — hot-path safe)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class RotationQuaternion:
    """Unit quaternion for a single 3D rotation.

    Auto-normalizes on construction so that arithmetic drift is contained
    within each operation rather than accumulating across frames. Raises
    ``ValueError`` on a near-zero input to prevent silently producing NaN
    axes from degenerate quaternions.

    Parameters
    ----------
    w, x, y, z : float
        Scalar and vector components. The input does *not* need to be
        pre-normalized; ``__post_init__`` handles it.
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            raise ValueError(
                f"Cannot normalize near-zero quaternion "
                f"(norm={norm:.2e}, w={self.w}, x={self.x}, y={self.y}, z={self.z})"
            )
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

    @classmethod
    def identity(cls) -> "RotationQuaternion":
        """Return the identity rotation ``(1, 0, 0, 0)``."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    # ── Basic operations ──────────────────────────────────────────

    def conjugate(self) -> "RotationQuaternion":
        """Return the conjugate ``(w, -x, -y, -z)``.

        For a unit quaternion the conjugate equals the inverse.
        """
        return RotationQuaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def inverse(self) -> "RotationQuaternion":
        """Return the inverse rotation. Same as ``conjugate`` for unit quaternions."""
        return self.conjugate()

    def __mul__(self, other: "RotationQuaternion") -> "RotationQuaternion":
        """Hamilton product ``self * other`` — composes the two rotations.

        The rotation represented by ``self * other`` is equivalent to
        applying ``other`` first, then ``self``: ``R(q₁·q₂) = R(q₁) ∘ R(q₂)``.
        """
        return RotationQuaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def dot(self, other: "RotationQuaternion") -> float:
        """Dot product ``w₁w₂ + x₁x₂ + y₁y₂ + z₁z₂``.

        For unit quaternions this is cos(θ/2) where θ is the rotation angle
        between them.
        """
        return (
            self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

    # ── Conversion: quaternion → other representations ────────────

    def to_rotation_matrix(self) -> NDArray[float64]:
        """Convert to a 3×3 rotation matrix.

        Returns a right-handed rotation matrix R such that for any vector
        v, ``R @ v`` rotates v by this quaternion.

        The formula is the standard one (see Diebel 2006 eq. 125):
            R = I + 2w·[u]× + 2[u]×²
        where u = (x, y, z) and [u]× is the cross-product matrix.
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_rotation_matrix(cls, R: NDArray[float64]) -> "RotationQuaternion":
        """Construct a quaternion from a 3×3 rotation matrix.

        Uses Shepperd's trace-based method (Shepperd 1978) which selects
        the numerically most stable branch based on the largest diagonal
        element of R.
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(
                f"Rotation matrix must be 3×3, got shape {R.shape}"
            )

        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return cls(
                w=0.25 / s,
                x=(R[2, 1] - R[1, 2]) * s,
                y=(R[0, 2] - R[2, 0]) * s,
                z=(R[1, 0] - R[0, 1]) * s,
            )
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return cls(
                w=(R[2, 1] - R[1, 2]) / s,
                x=0.25 * s,
                y=(R[0, 1] + R[1, 0]) / s,
                z=(R[0, 2] + R[2, 0]) / s,
            )
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return cls(
                w=(R[0, 2] - R[2, 0]) / s,
                x=(R[0, 1] + R[1, 0]) / s,
                y=0.25 * s,
                z=(R[1, 2] + R[2, 1]) / s,
            )
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            return cls(
                w=(R[1, 0] - R[0, 1]) / s,
                x=(R[0, 2] + R[2, 0]) / s,
                y=(R[1, 2] + R[2, 1]) / s,
                z=0.25 * s,
            )

    def to_axis_angle(self) -> tuple[NDArray[float64], float]:
        """Convert to axis-angle representation.

        Returns ``(axis, angle)`` where ``axis`` is a unit (3,) vector
        and ``angle`` is in radians in [0, π].
        """
        w_clamped = float(np.clip(self.w, -1.0, 1.0))
        angle = 2.0 * np.arccos(abs(w_clamped))
        sin_half = np.sqrt(1.0 - w_clamped**2)

        if sin_half < 1e-10:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0

        axis = np.array([self.x, self.y, self.z], dtype=np.float64) / sin_half
        if self.w < 0.0:
            axis = -axis
        return axis, angle

    def to_euler_xyz(self) -> tuple[float, float, float]:
        """Return ``(roll, pitch, yaw)`` in radians.

        Convention: **ZYX intrinsic** (aerospace convention).
        Equivalent to XYZ extrinsic. Rotation order: yaw around Z, then
        pitch around Y', then roll around X'' in the body frame.

        Formulas from Diebel (2006) eq. 356–358.
        """
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = float(np.arcsin(sinp))

        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))

        return roll, pitch, yaw

    def rotate_vector(self, v: NDArray[float64]) -> NDArray[float64]:
        """Rotate a (3,) vector by this quaternion.

        Uses the efficient Rodrigues-form formula:
            v' = v + 2w·(u × v) + 2·(u × (u × v))
        where u = (x, y, z). Avoids constructing the full 3×3 matrix.
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(f"Vector must have shape (3,), got {v.shape}")
        u = np.array([self.x, self.y, self.z], dtype=np.float64)
        uv = np.cross(u, v)
        uuv = np.cross(u, uv)
        return v + 2.0 * (self.w * uv + uuv)

    # ── Interpolation ─────────────────────────────────────────────

    @classmethod
    def slerp(
        cls, q0: "RotationQuaternion", q1: "RotationQuaternion", t: float
    ) -> "RotationQuaternion":
        """Spherical linear interpolation between two quaternions.

        Follows Shoemake (1985). Handles the double-cover ambiguity
        (q and −q represent the same rotation) by selecting the shorter
        arc. Falls back to normalized linear interpolation when the
        quaternions are nearly parallel (dot > 0.9995) to avoid division
        by a vanishing sin(θ).

        Parameters
        ----------
        q0 : RotationQuaternion
            Start rotation (t = 0).
        q1 : RotationQuaternion
            End rotation (t = 1).
        t : float
            Interpolation parameter in [0, 1].

        Returns
        -------
        RotationQuaternion
            Interpolated unit quaternion.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError(
                f"SLERP parameter t must be in [0, 1], got {t}"
            )

        dot = q0.dot(q1)

        # Take shorter arc on the hypersphere
        if dot < 0.0:
            q1_w, q1_x, q1_y, q1_z = -q1.w, -q1.x, -q1.y, -q1.z
            dot = -dot
        else:
            q1_w, q1_x, q1_y, q1_z = q1.w, q1.x, q1.y, q1.z

        dot = min(dot, 1.0)

        # Near-parallel → NLERP (avoid sin(θ) ≈ 0)
        if dot > 0.9995:
            return cls(
                w=q0.w + t * (q1_w - q0.w),
                x=q0.x + t * (q1_x - q0.x),
                y=q0.y + t * (q1_y - q0.y),
                z=q0.z + t * (q1_z - q0.z),
            )

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t

        s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0

        return cls(
            w=float(s0 * q0.w + s1 * q1_w),
            x=float(s0 * q0.x + s1 * q1_x),
            y=float(s0 * q0.y + s1 * q1_y),
            z=float(s0 * q0.z + s1 * q1_z),
        )


# ═══════════════════════════════════════════════════════════════════════
# Vectorized operations — pure numpy functions on (N, 4) arrays
# ═══════════════════════════════════════════════════════════════════════
#
# All functions operate on (N, 4) float64 arrays with columns [w, x, y, z].
# They are the vectorized equivalents of the RotationQuaternion methods above and
# are the ones used in the per-frame hot loop (orientation solver, rigid
# body kinematics). No per-element RotationQuaternion objects are created.


def normalize_quaternion_array(
    q: NDArray[float64],
) -> NDArray[float64]:
    """Normalize each row of an (N, 4) quaternion array to unit length.

    Returns a **new** array; the input is not modified.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) array, got shape {q.shape}"
        )
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        bad = np.where(norms.ravel() < 1e-10)[0]
        raise ValueError(
            f"Near-zero quaternion(s) at indices {bad.tolist()}"
        )
    return q / norms


def hamilton_product(
    q_a: NDArray[float64],
    q_b: NDArray[float64],
) -> NDArray[float64]:
    """Batch Hamilton product ``q_a * q_b`` for (N, 4) arrays.

    The result represents composing the rotation of ``q_b`` followed by
    the rotation of ``q_a``, matching the scalar ``RotationQuaternion.__mul__``
    semantics: ``R(q_a·q_b) = R(q_a) ∘ R(q_b)``.
    """
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    _check_quat_shape(q_a)
    _check_quat_shape(q_b)

    w1, x1, y1, z1 = q_a[:, 0], q_a[:, 1], q_a[:, 2], q_a[:, 3]
    w2, x2, y2, z2 = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]

    return np.column_stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def conjugate_quaternion_array(
    q: NDArray[float64],
) -> NDArray[float64]:
    """Return the conjugate of each quaternion in an (N, 4) array."""
    q = np.asarray(q, dtype=np.float64)
    _check_quat_shape(q)
    conj = q.copy()
    conj[:, 1:] *= -1.0
    return conj


def quaternion_to_rotation_matrix(
    q: NDArray[float64],
) -> NDArray[float64]:
    """Convert (N, 4) quaternion array to (N, 3, 3) rotation matrices.

    Same formula as ``RotationQuaternion.to_rotation_matrix``, vectorized.
    """
    q = np.asarray(q, dtype=np.float64)
    _check_quat_shape(q)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    n = len(q)
    R = np.empty((n, 3, 3), dtype=np.float64)

    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


def quaternion_to_axis_angle(
    q: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Convert (N, 4) quaternions to axis-angle.

    Returns ``(axes, angles)`` where ``axes`` is (N, 3) unit vectors and
    ``angles`` is (N,) in radians.
    """
    q = np.asarray(q, dtype=np.float64)
    _check_quat_shape(q)

    w = q[:, 0]
    xyz = q[:, 1:4]

    w_clamped = np.clip(w, -1.0, 1.0)
    angles = 2.0 * np.arccos(np.abs(w_clamped))
    sin_half = np.sqrt(1.0 - w_clamped**2)

    axes = np.zeros_like(xyz)
    valid = sin_half > 1e-10
    axes[valid] = xyz[valid] / sin_half[valid, np.newaxis]
    axes[~valid] = [1.0, 0.0, 0.0]

    # Flip axis when w < 0 (the quaternion on the opposite hemisphere
    # represents the same rotation)
    axes[w < 0] *= -1.0

    return axes, angles


def quaternion_to_euler(
    q: NDArray[float64],
) -> NDArray[float64]:
    """Convert (N, 4) quaternions to (N, 3) Euler angles.

    Returns columns ``[roll, pitch, yaw]`` in radians, ZYX intrinsic
    (aerospace) convention. Same formulas as ``RotationQuaternion.to_euler_xyz``.
    """
    q = np.asarray(q, dtype=np.float64)
    _check_quat_shape(q)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (around X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (around Y)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw (around Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([roll, pitch, yaw])


def rotate_vector_batch(
    q: NDArray[float64],
    v: NDArray[float64],
) -> NDArray[float64]:
    """Rotate a single (3,) vector by N quaternions.

    Returns an (N, 3) array where row i is **v** rotated by **q[i]**.
    Uses the Rodrigues form for efficiency.
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    _check_quat_shape(q)
    if v.shape != (3,):
        raise ValueError(f"Vector must have shape (3,), got {v.shape}")

    w = q[:, 0]          # (N,)
    u = q[:, 1:4]        # (N, 3)

    uv = np.cross(u, v)         # (N, 3)
    uuv = np.cross(u, uv)       # (N, 3)

    return v + 2.0 * w[:, np.newaxis] * uv + 2.0 * uuv


def rotate_vectors_batch(
    q: NDArray[float64],
    vectors: NDArray[float64],
) -> NDArray[float64]:
    """Rotate M vectors by N quaternions.

    Returns an (N, M, 3) array where ``result[n, m]`` is ``vectors[m]``
    rotated by ``q[n]``.
    """
    q = np.asarray(q, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    _check_quat_shape(q)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(
            f"vectors must have shape (M, 3), got {vectors.shape}"
        )

    n_frames = len(q)
    n_vectors = len(vectors)

    w = q[:, 0]                                    # (N,)
    u = q[:, 1:4]                                  # (N, 3)

    v_bc = np.broadcast_to(vectors, (n_frames, n_vectors, 3))
    u_bc = u[:, np.newaxis, :]                     # (N, 1, 3) → broadcasts

    uv = np.cross(u_bc, v_bc)                      # (N, M, 3)
    uuv = np.cross(u_bc, uv)

    return v_bc + 2.0 * w[:, np.newaxis, np.newaxis] * uv + 2.0 * uuv


# ── SLERP (vectorized) ──────────────────────────────────────────────


def slerp_batch(
    q0: NDArray[float64],
    q1: NDArray[float64],
    t: NDArray[float64],
) -> NDArray[float64]:
    """Vectorized SLERP between two arrays of quaternions.

    Parameters
    ----------
    q0 : (M, 4)
        Start quaternions.
    q1 : (M, 4)
        End quaternions.
    t : (M,)
        Interpolation parameters in [0, 1].

    Returns
    -------
    (M, 4)
        Interpolated quaternions, normalized.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    _check_quat_shape(q0)
    _check_quat_shape(q1)

    if t.ndim != 1 or len(t) != len(q0):
        raise ValueError(
            f"t must be (M,) matching q0 length, got shape {t.shape}"
        )

    # Double-cover: negate q1 where dot(q0, q1) < 0 to take shorter arc
    dot = np.sum(q0 * q1, axis=1)  # (M,)
    q1_adj = q1.copy()
    neg_mask = dot < 0.0
    q1_adj[neg_mask] = -q1_adj[neg_mask]
    dot = np.abs(dot)
    dot = np.clip(dot, 0.0, 1.0)

    result = np.empty_like(q0)

    # Near-parallel → NLERP (~0.002° threshold, avoids sin(θ)≈0)
    near_mask = dot > (1.0 - 1e-10)
    if np.any(near_mask):
        lerp = q0[near_mask] + t[near_mask, np.newaxis] * (
            q1_adj[near_mask] - q0[near_mask]
        )
        norms = np.linalg.norm(lerp, axis=1, keepdims=True)
        result[near_mask] = lerp / np.maximum(norms, 1e-10)

    # Full SLERP for the rest
    far_mask = ~near_mask
    if np.any(far_mask):
        theta = np.arccos(dot[far_mask])
        sin_theta = np.sin(theta)
        s0 = np.sin((1.0 - t[far_mask]) * theta) / sin_theta
        s1 = np.sin(t[far_mask] * theta) / sin_theta
        slerp_result = (
            s0[:, np.newaxis] * q0[far_mask]
            + s1[:, np.newaxis] * q1_adj[far_mask]
        )
        norms = np.linalg.norm(slerp_result, axis=1, keepdims=True)
        result[far_mask] = slerp_result / np.maximum(norms, 1e-10)

    return result


def slerp_resample(
    quaternions: NDArray[float64],
    original_timestamps: NDArray[float64],
    target_timestamps: NDArray[float64],
) -> NDArray[float64]:
    """Resample a quaternion trajectory to new timestamps via SLERP.

    Parameters
    ----------
    quaternions : (N, 4)
        Source quaternion trajectory.
    original_timestamps : (N,)
        Strictly increasing timestamps for the source frames.
    target_timestamps : (M,)
        Monotonically increasing target timestamps.

    Returns
    -------
    (M, 4)
        Interpolated quaternions at each target timestamp. Target times
        outside the original range are clamped to the boundary quaternion.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    original_timestamps = np.asarray(original_timestamps, dtype=np.float64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)

    _check_quat_shape(quaternions)
    n_original = len(quaternions)

    if len(original_timestamps) != n_original:
        raise ValueError(
            f"original_timestamps length ({len(original_timestamps)}) "
            f"must match quaternions length ({n_original})"
        )
    if n_original < 2:
        raise ValueError("Need at least 2 quaternions to resample")
    if not np.all(np.diff(original_timestamps) > 0):
        raise ValueError(
            "original_timestamps must be strictly increasing"
        )

    n_target = len(target_timestamps)
    indices = np.searchsorted(original_timestamps, target_timestamps)
    idx_lo = np.clip(indices - 1, 0, n_original - 2)
    idx_hi = idx_lo + 1

    q0 = quaternions[idx_lo]
    q1 = quaternions[idx_hi]
    t0 = original_timestamps[idx_lo]
    t1 = original_timestamps[idx_hi]

    dt = t1 - t0
    dt_safe = np.where(dt > 1e-10, dt, 1.0)
    t_param = (target_timestamps - t0) / dt_safe
    t_param = np.clip(t_param, 0.0, 1.0)
    # Clamp to boundaries for out-of-range target times
    t_param = np.where(dt > 1e-10, t_param, 0.0)

    return slerp_batch(q0, q1, t_param)


# ── Angular velocity ───────────────────────────────────────────────


def compute_angular_velocity(
    quaternions: NDArray[float64],
    timestamps: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute angular velocity from a quaternion trajectory.

    Uses finite differences: forward at frame 0, central for interior
    frames, backward at the final frame. Relative quaternion between
    consecutive poses is converted to axis-angle, then ω = axis · (θ / Δt).

    Parameters
    ----------
    quaternions : (N, 4)
    timestamps : (N,)
        Strictly increasing timestamps in seconds.

    Returns
    -------
    omega_global : (N, 3)
        Angular velocity in the **world** frame (rad/s).
    omega_local : (N, 3)
        Angular velocity in the **body** frame (rad/s). Computed as
        Rᵀ @ ω_global where R is the rotation matrix for each frame.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _check_quat_shape(quaternions)

    n = len(quaternions)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    dt_arr = np.diff(timestamps)
    if np.any(dt_arr <= 1e-10):
        bad = np.where(dt_arr <= 1e-10)[0]
        raise ValueError(
            f"Timestamps must be strictly increasing; "
            f"bad dt at frame {bad[0]} → {bad[0] + 1}: {dt_arr[bad[0]]:.2e}"
        )

    # Build index pairs for finite differences
    # Frame 0:      forward  (curr=0,     next=1)
    # Frames 1..N-2: central  (curr=i-1,   next=i+1)
    # Frame N-1:    backward (curr=N-2,   next=N-1)
    idx_curr = np.concatenate(
        [[0], np.arange(0, n - 2), [n - 2]]
    )
    idx_next = np.concatenate(
        [[1], np.arange(2, n), [n - 1]]
    )

    time_deltas = np.empty(n, dtype=np.float64)
    time_deltas[0] = timestamps[1] - timestamps[0]
    time_deltas[1:-1] = timestamps[2:] - timestamps[:-2]
    time_deltas[-1] = timestamps[-1] - timestamps[-2]

    # Relative quaternion: q_rel = q_next * conj(q_curr)
    q_curr = quaternions[idx_curr]
    q_next = quaternions[idx_next]
    q_curr_conj = conjugate_quaternion_array(q_curr)
    q_rel = hamilton_product(q_next, q_curr_conj)

    axes, angles = quaternion_to_axis_angle(q_rel)
    omega_global = axes * (angles / time_deltas)[:, np.newaxis]

    # Local: transform by current orientation
    R = quaternion_to_rotation_matrix(quaternions)
    omega_local = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), omega_global)

    return omega_global, omega_local


# ── Composition ────────────────────────────────────────────────────


def compose_with_constant(
    quaternions: NDArray[float64],
    constant: NDArray[float64],
    *,
    pre_multiply: bool = True,
) -> NDArray[float64]:
    """Compose every quaternion in an (N, 4) trajectory with a constant.

    Parameters
    ----------
    quaternions : (N, 4)
    constant : (4,)
    pre_multiply : bool
        If True, result = constant * quaternions (constant applied second).
        If False, result = quaternions * constant (constant applied first).

    Returns
    -------
    (N, 4)
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    constant = np.asarray(constant, dtype=np.float64)
    _check_quat_shape(quaternions)
    if constant.shape != (4,):
        raise ValueError(
            f"constant must have shape (4,), got {constant.shape}"
        )
    q_const = np.broadcast_to(constant, (len(quaternions), 4))
    if pre_multiply:
        return hamilton_product(q_const, quaternions)
    else:
        return hamilton_product(quaternions, q_const)


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _check_quat_shape(q: NDArray[float64]) -> None:
    """Raise ValueError if *q* is not (N, 4)."""
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) quaternion array, got shape {q.shape}"
        )
