"""RotationQuaternion algebra for 3D rotations - scalar and vectorized operations.

This is the single home for all quaternion math. Every operation lives here exactly once:
the scalar ``RotationQuaternion`` dataclass for single-frame use, and pure numpy
module-level functions for batch operations on ``(N, 4)`` arrays. No other module in the
project should contain a Hamilton product, SLERP, or quaternion-to-matrix conversion.

The two halves are deliberate mirrors of each other: every batched function computes
exactly what the scalar method computes, element for element, using the same thresholds
from ``numeric_tolerances``. That equivalence is the contract, and it is what
``test_rotation_quaternion.py`` checks - so the batched half can be trusted in the hot
loop without being reasoned about separately.

Convention
----------
- Scalar-first ordering: **[w, x, y, z]** everywhere - on the wire, in numpy arrays, and
  in ``RotationQuaternion`` field order.
- All quaternions are **unit** quaternions representing rotations. The scalar
  ``RotationQuaternion`` is frozen and validates unit-ness on construction;
  ``RotationQuaternion.from_components`` normalizes first for callers holding raw numbers.
  Vectorized functions assume pre-normalized input (call ``normalize_quaternion_array``
  if needed).
- Identity rotation is ``(1, 0, 0, 0)`` - this is the T-pose contract for every bone in
  the standard human model.
- ``q`` and ``-q`` are the same rotation. Nothing here compares quaternions by value;
  use :meth:`RotationQuaternion.is_same_rotation`, which knows that.

References
----------
- Hamilton (1844) "On Quaternions" - Hamilton product definition.
- Shoemake (1985) "Animating Rotation with Quaternion Curves" - SLERP.
- Shepperd (1978) "Quaternion from Rotation Matrix" - trace-based matrix-to-quaternion
  in ``from_rotation_matrix``.
- Diebel (2006) "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation
  Vectors" - the ZYX intrinsic roll/pitch/yaw decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_QUATERNION_NORM,
    MINIMUM_QUATERNION_SINE,
    MINIMUM_SLERP_SEPARATION_COSINE,
    MINIMUM_TIME_DELTA_SECONDS,
    MINIMUM_VECTOR_NORM,
    UNIT_QUATERNION_TOLERANCE,
)
from skellyforge.type_overloads import FloatArray

NUMBER_OF_QUATERNION_COMPONENTS: int = 4
NUMBER_OF_SPATIAL_DIMENSIONS: int = 3


# ═══════════════════════════════════════════════════════════════════════
# Scalar RotationQuaternion (slot-based dataclass - hot-path safe)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(slots=True, frozen=True, eq=False)
class RotationQuaternion:
    """Unit quaternion for a single 3D rotation.

    Immutable once constructed: ``__post_init__`` validates that the components are
    finite and unit-length, and raises otherwise. Normalization happens *before*
    construction, in :meth:`from_components`, so that arithmetic drift is contained
    within each operation without any post-construction mutation of a frozen dataclass.
    Every operation here that can drift off the unit sphere routes its result through
    :meth:`from_components`.

    Value equality is deliberately disabled - matching every other value type in the
    geometry package, and for a stronger reason here: ``q`` and ``-q`` are the same
    rotation, so component-wise equality answers the wrong question. Ask
    :meth:`is_same_rotation` instead.

    Parameters
    ----------
    w, x, y, z : float
        Scalar and vector components of a **unit** quaternion. Construction validates
        unit-ness and raises otherwise; it never rewrites what it was given. Build from
        unnormalized components with :meth:`from_components`, which normalizes first.
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(component) for component in (self.w, self.x, self.y, self.z)
        ):
            raise ValueError(
                f"Quaternion components must be finite - got "
                f"(w={self.w}, x={self.x}, y={self.y}, z={self.z})"
            )
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if abs(norm - 1.0) > UNIT_QUATERNION_TOLERANCE:
            raise ValueError(
                f"RotationQuaternion must be unit length - got norm {norm:.12f} from "
                f"(w={self.w}, x={self.x}, y={self.y}, z={self.z}). Use "
                f"RotationQuaternion.from_components(...) to normalize before constructing."
            )

    @classmethod
    def from_components(
        cls, *, w: float, x: float, y: float, z: float
    ) -> RotationQuaternion:
        """Normalize the given components, then construct.

        This is the only path that accepts unnormalized input. Normalizing here, ahead of
        ``__init__``, is what lets the dataclass stay frozen: the instance is unit-length
        from the moment it exists and is never rewritten afterwards.
        """
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        if norm < MINIMUM_QUATERNION_NORM:
            raise ValueError(
                f"Cannot normalize near-zero quaternion "
                f"(norm={norm:.2e}, w={w}, x={x}, y={y}, z={z})"
            )
        return cls(w=w / norm, x=x / norm, y=y / norm, z=z / norm)

    @classmethod
    def from_array(cls, *, array: FloatArray) -> RotationQuaternion:
        """Build from a ``(4,)`` ``[w, x, y, z]`` array, normalizing first."""
        array = np.asarray(array, dtype=np.float64)
        if array.shape != (NUMBER_OF_QUATERNION_COMPONENTS,):
            raise ValueError(
                f"A quaternion array must have shape "
                f"({NUMBER_OF_QUATERNION_COMPONENTS},), got {array.shape}"
            )
        return cls.from_components(
            w=float(array[0]), x=float(array[1]), y=float(array[2]), z=float(array[3])
        )

    @classmethod
    def identity(cls) -> RotationQuaternion:
        """The identity rotation ``(1, 0, 0, 0)``."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    # ── Basic operations ──────────────────────────────────────────

    def as_array(self) -> FloatArray:
        """This rotation as a ``(4,)`` ``[w, x, y, z]`` array."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    def conjugate(self) -> RotationQuaternion:
        """The conjugate ``(w, -x, -y, -z)``.

        For a unit quaternion the conjugate equals the inverse.
        """
        return RotationQuaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def inverse(self) -> RotationQuaternion:
        """The inverse rotation. Same as :meth:`conjugate` for unit quaternions."""
        return self.conjugate()

    def __mul__(self, other: RotationQuaternion) -> RotationQuaternion:
        """Hamilton product ``self * other`` - composes the two rotations.

        The rotation represented by ``self * other`` is equivalent to applying ``other``
        first, then ``self``: ``R(q1 . q2) = R(q1) o R(q2)``.
        """
        if not isinstance(other, RotationQuaternion):
            return NotImplemented
        return RotationQuaternion.from_components(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def dot(self, *, other: RotationQuaternion) -> float:
        """Dot product ``w1w2 + x1x2 + y1y2 + z1z2``.

        For unit quaternions this is ``cos(theta/2)`` where ``theta`` is the rotation
        angle between them. Its SIGN carries only which side of the double cover the two
        happen to be written on, so take ``abs`` before comparing rotations.
        """
        return (
            self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

    def angle_to(self, *, other: RotationQuaternion) -> float:
        """The shortest-arc angle in radians between the two rotations, in ``[0, pi]``.

        Measured on the relative rotation rather than as ``2 * arccos(|dot|)``. Both are
        the same in exact arithmetic, but ``arccos`` has a vertical tangent at 1 and two
        nearly-equal rotations land there: a dot one ulp below 1 comes back as an angle of
        1e-8 rather than 0, which is a hundred times the tolerance anything would want to
        compare against.
        """
        return (self.inverse() * other).to_axis_angle()[1]

    def is_same_rotation(
        self, *, other: RotationQuaternion, tolerance_radians: float = 1e-9
    ) -> bool:
        """Whether two quaternions denote the same rotation, double cover included."""
        return self.angle_to(other=other) <= tolerance_radians

    # ── Conversion: quaternion → other representations ────────────

    def to_rotation_matrix(self) -> FloatArray:
        """Convert to a 3x3 rotation matrix.

        Returns a right-handed rotation matrix ``R`` such that for any vector ``v``,
        ``R @ v`` rotates ``v`` by this quaternion.

        The formula is the standard one (Diebel 2006 eq. 125):
        ``R = I + 2w[u]x + 2[u]x^2`` where ``u = (x, y, z)``.
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
    def from_rotation_matrix(cls, *, matrix: FloatArray) -> RotationQuaternion:
        """Construct a quaternion from a 3x3 rotation matrix.

        Uses Shepperd's trace-based method (Shepperd 1978), which selects the numerically
        most stable branch based on the largest diagonal element.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (NUMBER_OF_SPATIAL_DIMENSIONS, NUMBER_OF_SPATIAL_DIMENSIONS):
            raise ValueError(f"Rotation matrix must be 3x3, got shape {matrix.shape}")

        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

        if trace > 0.0:
            scale = 0.5 / np.sqrt(trace + 1.0)
            return cls.from_components(
                w=0.25 / scale,
                x=(matrix[2, 1] - matrix[1, 2]) * scale,
                y=(matrix[0, 2] - matrix[2, 0]) * scale,
                z=(matrix[1, 0] - matrix[0, 1]) * scale,
            )
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            return cls.from_components(
                w=(matrix[2, 1] - matrix[1, 2]) / scale,
                x=0.25 * scale,
                y=(matrix[0, 1] + matrix[1, 0]) / scale,
                z=(matrix[0, 2] + matrix[2, 0]) / scale,
            )
        if matrix[1, 1] > matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            return cls.from_components(
                w=(matrix[0, 2] - matrix[2, 0]) / scale,
                x=(matrix[0, 1] + matrix[1, 0]) / scale,
                y=0.25 * scale,
                z=(matrix[1, 2] + matrix[2, 1]) / scale,
            )
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        return cls.from_components(
            w=(matrix[1, 0] - matrix[0, 1]) / scale,
            x=(matrix[0, 2] + matrix[2, 0]) / scale,
            y=(matrix[1, 2] + matrix[2, 1]) / scale,
            z=0.25 * scale,
        )

    def to_axis_angle(self) -> tuple[FloatArray, float]:
        """Convert to axis-angle.

        Returns ``(axis, angle)`` where ``axis`` is a unit ``(3,)`` vector and ``angle``
        is in radians in ``[0, pi]`` - the shortest arc, since ``q`` and ``-q`` are the
        same rotation.
        """
        # The vector part's norm IS sin(theta/2) for a unit quaternion, so taking it
        # directly beats recovering it as sqrt(1 - w^2), which cancels catastrophically
        # once w approaches 1 - exactly where small rotations live. Pairing it with
        # `atan2` keeps the angle accurate across the whole range.
        vector_part = np.array([self.x, self.y, self.z], dtype=np.float64)
        half_angle_sine = float(np.linalg.norm(vector_part))
        angle = float(2.0 * np.arctan2(half_angle_sine, abs(self.w)))

        if half_angle_sine < MINIMUM_QUATERNION_SINE:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0

        axis = vector_part / half_angle_sine
        if self.w < 0.0:
            axis = -axis
        return axis, angle

    @classmethod
    def from_rotation_vector(cls, *, rotation_vector: FloatArray) -> RotationQuaternion:
        """Exponential map: a rotation vector (axis x angle) to a quaternion.

        The inverse of :meth:`to_rotation_vector`. The vector's magnitude is the rotation
        angle in radians; its direction is the rotation axis. Uses the small-angle limit
        near zero, where ``sin(theta/2)/theta`` approaches ``1/2`` but is numerically
        unstable evaluated directly.
        """
        rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
        if rotation_vector.shape != (NUMBER_OF_SPATIAL_DIMENSIONS,):
            raise ValueError(
                f"Rotation vector must have shape ({NUMBER_OF_SPATIAL_DIMENSIONS},), "
                f"got {rotation_vector.shape}"
            )

        angle = float(np.linalg.norm(rotation_vector))
        if angle < MINIMUM_VECTOR_NORM:
            half = 0.5 * rotation_vector
            return cls.from_components(
                w=1.0, x=float(half[0]), y=float(half[1]), z=float(half[2])
            )

        axis = rotation_vector / angle
        half_angle_sine = float(np.sin(angle / 2.0))
        return cls.from_components(
            w=float(np.cos(angle / 2.0)),
            x=float(axis[0] * half_angle_sine),
            y=float(axis[1] * half_angle_sine),
            z=float(axis[2] * half_angle_sine),
        )

    def to_rotation_vector(self) -> FloatArray:
        """Logarithmic map: this rotation as a vector (axis x angle).

        The inverse of :meth:`from_rotation_vector`. Magnitude is the angle in radians in
        ``[0, pi]``. This is the tangent-space representation orientation filters work in:
        rotations are not a vector space, but rotation vectors are.
        """
        axis, angle = self.to_axis_angle()
        return axis * angle

    def to_roll_pitch_yaw(self) -> tuple[float, float, float]:
        """Return ``(roll, pitch, yaw)`` in radians, ZYX intrinsic (aerospace).

        Yaw about z, then pitch about y', then roll about x'' in the body frame. Named for
        what it returns rather than for an axis-letter ordering, because the two orderings
        that describe this convention (ZYX intrinsic, XYZ extrinsic) are the same thing
        and naming it after either one invites the reader to assume the other.

        Formulas from Diebel (2006) eq. 356-358.
        """
        roll_sine_cosine = 2.0 * (self.w * self.x + self.y * self.z)
        roll_cosine_cosine = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = float(np.arctan2(roll_sine_cosine, roll_cosine_cosine))

        pitch_sine = float(np.clip(2.0 * (self.w * self.y - self.z * self.x), -1.0, 1.0))
        pitch = float(np.arcsin(pitch_sine))

        yaw_sine_cosine = 2.0 * (self.w * self.z + self.x * self.y)
        yaw_cosine_cosine = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = float(np.arctan2(yaw_sine_cosine, yaw_cosine_cosine))

        return roll, pitch, yaw

    def rotate_vector(self, *, vector: FloatArray) -> FloatArray:
        """Rotate a ``(3,)`` vector by this quaternion.

        Uses the Rodrigues form ``v' = v + 2w(u x v) + 2(u x (u x v))`` where
        ``u = (x, y, z)``, avoiding the full 3x3 matrix.
        """
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (NUMBER_OF_SPATIAL_DIMENSIONS,):
            raise ValueError(
                f"Vector must have shape ({NUMBER_OF_SPATIAL_DIMENSIONS},), "
                f"got {vector.shape}"
            )
        axis_part = np.array([self.x, self.y, self.z], dtype=np.float64)
        cross_once = np.cross(axis_part, vector)
        cross_twice = np.cross(axis_part, cross_once)
        return vector + 2.0 * (self.w * cross_once + cross_twice)

    # ── Interpolation ─────────────────────────────────────────────

    @classmethod
    def slerp(
        cls,
        *,
        start: RotationQuaternion,
        end: RotationQuaternion,
        fraction: float,
    ) -> RotationQuaternion:
        """Spherical linear interpolation between two rotations.

        Follows Shoemake (1985). Handles the double-cover ambiguity by taking the shorter
        arc, and falls back to normalized linear interpolation once the two are closer
        together than ``MINIMUM_SLERP_SEPARATION_COSINE``, where ``sin(theta)`` vanishes.
        That threshold is shared with :func:`slerp_batch`, so the scalar and batched
        results agree by construction rather than by coincidence.

        Args:
            start: the rotation at ``fraction = 0``.
            end: the rotation at ``fraction = 1``.
            fraction: interpolation parameter in ``[0, 1]``.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"SLERP fraction must be in [0, 1], got {fraction}")

        cosine = start.dot(other=end)
        # Take the shorter arc on the hypersphere.
        if cosine < 0.0:
            end_w, end_x, end_y, end_z = -end.w, -end.x, -end.y, -end.z
            cosine = -cosine
        else:
            end_w, end_x, end_y, end_z = end.w, end.x, end.y, end.z
        cosine = min(cosine, 1.0)

        if cosine > MINIMUM_SLERP_SEPARATION_COSINE:
            return cls.from_components(
                w=start.w + fraction * (end_w - start.w),
                x=start.x + fraction * (end_x - start.x),
                y=start.y + fraction * (end_y - start.y),
                z=start.z + fraction * (end_z - start.z),
            )

        total_angle = np.arccos(cosine)
        total_angle_sine = np.sin(total_angle)
        start_weight = np.sin((1.0 - fraction) * total_angle) / total_angle_sine
        end_weight = np.sin(fraction * total_angle) / total_angle_sine

        return cls.from_components(
            w=float(start_weight * start.w + end_weight * end_w),
            x=float(start_weight * start.x + end_weight * end_x),
            y=float(start_weight * start.y + end_weight * end_y),
            z=float(start_weight * start.z + end_weight * end_z),
        )

    def __repr__(self) -> str:
        return (
            f"RotationQuaternion(w={self.w:.6f}, x={self.x:.6f}, "
            f"y={self.y:.6f}, z={self.z:.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Vectorized operations - pure numpy functions on (N, 4) arrays
# ═══════════════════════════════════════════════════════════════════════
#
# All functions operate on (N, 4) float64 arrays with columns [w, x, y, z]. They are the
# vectorized equivalents of the RotationQuaternion methods above, element for element and
# threshold for threshold, and are the ones used in the per-frame hot loop. No per-element
# RotationQuaternion objects are created.


def normalize_quaternion_array(*, quaternions: FloatArray) -> FloatArray:
    """Normalize each row of an ``(N, 4)`` quaternion array to unit length.

    Returns a **new** array; the input is not modified.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < MINIMUM_QUATERNION_NORM):
        degenerate = np.where(norms.ravel() < MINIMUM_QUATERNION_NORM)[0]
        raise ValueError(f"Near-zero quaternion(s) at indices {degenerate.tolist()}")
    return quaternions / norms


def hamilton_product(*, left: FloatArray, right: FloatArray) -> FloatArray:
    """Batch Hamilton product ``left * right`` for ``(N, 4)`` arrays.

    Composes the rotation of ``right`` followed by the rotation of ``left``, matching the
    scalar ``RotationQuaternion.__mul__``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=left)
    _raise_unless_quaternion_array(quaternions=right)

    left_w, left_x, left_y, left_z = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    right_w, right_x, right_y, right_z = (
        right[:, 0],
        right[:, 1],
        right[:, 2],
        right[:, 3],
    )

    return np.column_stack(
        [
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ]
    )


def conjugate_quaternion_array(*, quaternions: FloatArray) -> FloatArray:
    """The conjugate of each quaternion in an ``(N, 4)`` array."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)
    conjugated = quaternions.copy()
    conjugated[:, 1:] *= -1.0
    return conjugated


def quaternions_to_rotation_matrices(*, quaternions: FloatArray) -> FloatArray:
    """Convert an ``(N, 4)`` quaternion array to ``(N, 3, 3)`` rotation matrices."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)

    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrices = np.empty(
        (len(quaternions), NUMBER_OF_SPATIAL_DIMENSIONS, NUMBER_OF_SPATIAL_DIMENSIONS),
        dtype=np.float64,
    )
    matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrices[:, 0, 1] = 2.0 * (xy - wz)
    matrices[:, 0, 2] = 2.0 * (xz + wy)
    matrices[:, 1, 0] = 2.0 * (xy + wz)
    matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrices[:, 1, 2] = 2.0 * (yz - wx)
    matrices[:, 2, 0] = 2.0 * (xz - wy)
    matrices[:, 2, 1] = 2.0 * (yz + wx)
    matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrices


def quaternions_to_axis_angles(
    *, quaternions: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Convert ``(N, 4)`` quaternions to axis-angle.

    Returns ``(axes, angles)`` where ``axes`` is ``(N, 3)`` unit vectors and ``angles`` is
    ``(N,)`` in radians, in ``[0, pi]``.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)

    scalar_parts = quaternions[:, 0]
    vector_parts = quaternions[:, 1:4]

    # Same reasoning as the scalar `to_axis_angle`: the vector part's norm is sin(theta/2)
    # exactly, and `atan2` against |w| stays accurate where `arccos` does not.
    half_angle_sines = np.linalg.norm(vector_parts, axis=1)
    angles = 2.0 * np.arctan2(half_angle_sines, np.abs(scalar_parts))

    axes = np.zeros_like(vector_parts)
    resolvable = half_angle_sines > MINIMUM_QUATERNION_SINE
    axes[resolvable] = (
        vector_parts[resolvable] / half_angle_sines[resolvable, np.newaxis]
    )
    axes[~resolvable] = [1.0, 0.0, 0.0]

    # A quaternion on the opposite hemisphere denotes the same rotation about the
    # opposite axis, so flip it to keep every angle in [0, pi].
    axes[scalar_parts < 0.0] *= -1.0

    return axes, angles


def quaternions_to_roll_pitch_yaw(*, quaternions: FloatArray) -> FloatArray:
    """Convert ``(N, 4)`` quaternions to ``(N, 3)`` ``[roll, pitch, yaw]`` in radians.

    ZYX intrinsic (aerospace), matching ``RotationQuaternion.to_roll_pitch_yaw``.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)

    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return np.column_stack([roll, pitch, yaw])


def rotate_vectors_batch(*, quaternions: FloatArray, vectors: FloatArray) -> FloatArray:
    """Rotate M vectors by N quaternions.

    Returns an ``(N, M, 3)`` array where ``result[n, m]`` is ``vectors[m]`` rotated by
    ``quaternions[n]``. Rotating a single vector is the ``M = 1`` case; there is no
    separate single-vector entry point, because there is no separate arithmetic.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)
    if vectors.ndim != 2 or vectors.shape[1] != NUMBER_OF_SPATIAL_DIMENSIONS:
        raise ValueError(
            f"vectors must have shape (M, {NUMBER_OF_SPATIAL_DIMENSIONS}), "
            f"got {vectors.shape}"
        )

    scalar_parts = quaternions[:, 0]
    vector_parts = quaternions[:, 1:4]

    broadcast_vectors = np.broadcast_to(
        vectors, (len(quaternions), len(vectors), NUMBER_OF_SPATIAL_DIMENSIONS)
    )
    broadcast_axes = vector_parts[:, np.newaxis, :]

    cross_once = np.cross(broadcast_axes, broadcast_vectors)
    cross_twice = np.cross(broadcast_axes, cross_once)

    return (
        broadcast_vectors
        + 2.0 * scalar_parts[:, np.newaxis, np.newaxis] * cross_once
        + 2.0 * cross_twice
    )


# ── SLERP (vectorized) ──────────────────────────────────────────────


def slerp_batch(
    *, start: FloatArray, end: FloatArray, fractions: FloatArray
) -> FloatArray:
    """Vectorized SLERP between two ``(M, 4)`` arrays of quaternions.

    Args:
        start: ``(M, 4)`` rotations at ``fraction = 0``.
        end: ``(M, 4)`` rotations at ``fraction = 1``.
        fractions: ``(M,)`` interpolation parameters in ``[0, 1]``.

    Returns:
        ``(M, 4)`` interpolated unit quaternions. Uses the same shorter-arc rule and the
        same ``MINIMUM_SLERP_SEPARATION_COSINE`` fallback as the scalar
        :meth:`RotationQuaternion.slerp`.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=start)
    _raise_unless_quaternion_array(quaternions=end)

    if fractions.ndim != 1 or len(fractions) != len(start):
        raise ValueError(
            f"fractions must be (M,) matching start's length {len(start)}, got shape "
            f"{fractions.shape}"
        )
    if np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise ValueError("SLERP fractions must all be in [0, 1]")

    # Double cover: negate `end` where the dot is negative, to take the shorter arc.
    cosines = np.sum(start * end, axis=1)
    shorter_arc_end = np.where(cosines[:, np.newaxis] < 0.0, -end, end)
    cosines = np.clip(np.abs(cosines), 0.0, 1.0)

    result = np.empty_like(start)

    near = cosines > MINIMUM_SLERP_SEPARATION_COSINE
    if np.any(near):
        interpolated = start[near] + fractions[near, np.newaxis] * (
            shorter_arc_end[near] - start[near]
        )
        result[near] = interpolated / np.linalg.norm(
            interpolated, axis=1, keepdims=True
        )

    far = ~near
    if np.any(far):
        total_angles = np.arccos(cosines[far])
        total_angle_sines = np.sin(total_angles)
        start_weights = np.sin((1.0 - fractions[far]) * total_angles) / total_angle_sines
        end_weights = np.sin(fractions[far] * total_angles) / total_angle_sines
        interpolated = (
            start_weights[:, np.newaxis] * start[far]
            + end_weights[:, np.newaxis] * shorter_arc_end[far]
        )
        result[far] = interpolated / np.linalg.norm(interpolated, axis=1, keepdims=True)

    return result


def slerp_resample(
    *,
    quaternions: FloatArray,
    original_timestamps: FloatArray,
    target_timestamps: FloatArray,
) -> FloatArray:
    """Resample a quaternion trajectory to new timestamps via SLERP.

    Args:
        quaternions: ``(N, 4)`` source trajectory.
        original_timestamps: ``(N,)`` strictly increasing timestamps for the source frames.
        target_timestamps: ``(M,)`` target timestamps.

    Returns:
        ``(M, 4)`` interpolated quaternions. Target times outside the source range are
        clamped to the boundary rotation.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    original_timestamps = np.asarray(original_timestamps, dtype=np.float64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)

    _raise_unless_quaternion_array(quaternions=quaternions)
    number_of_source_frames = len(quaternions)

    if len(original_timestamps) != number_of_source_frames:
        raise ValueError(
            f"original_timestamps length ({len(original_timestamps)}) must match "
            f"quaternions length ({number_of_source_frames})"
        )
    if number_of_source_frames < 2:
        raise ValueError("Need at least 2 quaternions to resample")
    _raise_unless_strictly_increasing(timestamps=original_timestamps)

    insertion_points = np.searchsorted(original_timestamps, target_timestamps)
    lower_indices = np.clip(insertion_points - 1, 0, number_of_source_frames - 2)
    upper_indices = lower_indices + 1

    lower_times = original_timestamps[lower_indices]
    upper_times = original_timestamps[upper_indices]
    # Strictly increasing timestamps were enforced above, so every span is positive and
    # the clip alone handles targets that fall outside the source range.
    fractions = np.clip(
        (target_timestamps - lower_times) / (upper_times - lower_times), 0.0, 1.0
    )

    return slerp_batch(
        start=quaternions[lower_indices],
        end=quaternions[upper_indices],
        fractions=fractions,
    )


# ── Angular velocity ───────────────────────────────────────────────


def compute_angular_velocity(
    *, quaternions: FloatArray, timestamps: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Angular velocity from a quaternion trajectory, by finite differences.

    Forward difference at frame 0, central for interior frames, backward at the last one.
    The relative rotation between the bracketing poses is converted to axis-angle, then
    ``omega = axis * (theta / dt)``.

    Args:
        quaternions: ``(N, 4)`` trajectory.
        timestamps: ``(N,)`` strictly increasing timestamps in seconds.

    Returns:
        ``(omega_world, omega_body)``, each ``(N, 3)`` in radians per second.
        ``omega_body`` is ``R^T @ omega_world`` for each frame's rotation.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)

    number_of_frames = len(quaternions)
    if number_of_frames < 2:
        raise ValueError(f"Need at least 2 frames, got {number_of_frames}")
    if len(timestamps) != number_of_frames:
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match quaternions length "
            f"({number_of_frames})"
        )
    _raise_unless_strictly_increasing(timestamps=timestamps)

    earlier_indices = np.concatenate(
        [[0], np.arange(0, number_of_frames - 2), [number_of_frames - 2]]
    )
    later_indices = np.concatenate(
        [[1], np.arange(2, number_of_frames), [number_of_frames - 1]]
    )

    time_deltas = timestamps[later_indices] - timestamps[earlier_indices]

    relative = hamilton_product(
        left=quaternions[later_indices],
        right=conjugate_quaternion_array(quaternions=quaternions[earlier_indices]),
    )
    axes, angles = quaternions_to_axis_angles(quaternions=relative)
    omega_world = axes * (angles / time_deltas)[:, np.newaxis]

    matrices = quaternions_to_rotation_matrices(quaternions=quaternions)
    omega_body = np.einsum("nij,nj->ni", matrices.transpose(0, 2, 1), omega_world)

    return omega_world, omega_body


# ── Composition ────────────────────────────────────────────────────


def pre_multiply_by_constant(
    *, quaternions: FloatArray, constant: FloatArray
) -> FloatArray:
    """``constant * quaternions`` for every row: the constant rotation applied SECOND."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)
    return hamilton_product(
        left=_broadcast_constant(constant=constant, count=len(quaternions)),
        right=quaternions,
    )


def post_multiply_by_constant(
    *, quaternions: FloatArray, constant: FloatArray
) -> FloatArray:
    """``quaternions * constant`` for every row: the constant rotation applied FIRST."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _raise_unless_quaternion_array(quaternions=quaternions)
    return hamilton_product(
        left=quaternions,
        right=_broadcast_constant(constant=constant, count=len(quaternions)),
    )


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _broadcast_constant(*, constant: FloatArray, count: int) -> FloatArray:
    """One ``(4,)`` quaternion repeated into an ``(N, 4)`` array."""
    constant = np.asarray(constant, dtype=np.float64)
    if constant.shape != (NUMBER_OF_QUATERNION_COMPONENTS,):
        raise ValueError(
            f"constant must have shape ({NUMBER_OF_QUATERNION_COMPONENTS},), "
            f"got {constant.shape}"
        )
    return np.broadcast_to(constant, (count, NUMBER_OF_QUATERNION_COMPONENTS))


def _raise_unless_quaternion_array(*, quaternions: FloatArray) -> None:
    """Raise unless `quaternions` is an ``(N, 4)`` array."""
    if quaternions.ndim != 2 or quaternions.shape[1] != NUMBER_OF_QUATERNION_COMPONENTS:
        raise ValueError(
            f"Expected (N, {NUMBER_OF_QUATERNION_COMPONENTS}) quaternion array, got "
            f"shape {quaternions.shape}"
        )


def _raise_unless_strictly_increasing(*, timestamps: FloatArray) -> None:
    """Raise unless every consecutive timestamp advances by a meaningful amount."""
    deltas = np.diff(timestamps)
    if np.any(deltas <= MINIMUM_TIME_DELTA_SECONDS):
        first_bad = int(np.where(deltas <= MINIMUM_TIME_DELTA_SECONDS)[0][0])
        raise ValueError(
            f"Timestamps must be strictly increasing; bad delta at frame {first_bad} -> "
            f"{first_bad + 1}: {deltas[first_bad]:.2e}"
        )
