"""Rigid-body kinematics — position + orientation → derived quantities.

Provides both a batch ``RigidBodyKinematics`` dataclass (for posthoc
analysis of full trajectories) and the module-level vectorized functions
it wraps (for direct use in the realtime hot loop, where constructing a
dataclass per bone per frame would be wasteful).

All derived quantities are **lazy ``cached_property``** on the batch model:
only the pose arrays (position + quaternion) and timestamps are stored;
velocity, acceleration, angular velocity, Euler angles, and keypoint
world positions are computed once on first access and cached.

Conventions
-----------
- Units: position in mm, velocity in mm/s, acceleration in mm/s²,
  angular velocity in rad/s, angular acceleration in rad/s².
- RotationQuaternion order: ``[w, x, y, z]`` — consistent with ``quaternion_math``
  and the frame message's channel layout.
- Derivatives use finite differences: forward at frame 0, central for
  interior frames, backward at the final frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.quaternion_math import (
    RotationQuaternion,
    compute_angular_velocity as _compute_angular_velocity,
    normalize_quaternion_array,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotate_vectors_batch,
    slerp_resample,
)

if TYPE_CHECKING:
    from numpy import float64
    from skellyforge.skellymodels.standard_human.reference_geometry import (
        SegmentReferenceGeometry,
    )


# ═══════════════════════════════════════════════════════════════════════
# Module-level vectorized functions (hot-loop safe)
# ═══════════════════════════════════════════════════════════════════════


def compute_linear_velocity(
    positions: NDArray[float64],
    timestamps: NDArray[float64],
) -> NDArray[float64]:
    """Compute linear velocity from positions via finite differences.

    Forward difference at frame 0, central for interior frames, backward
    at the final frame. Produces the same scheme as the angular velocity
    computation in ``quaternion_math.compute_angular_velocity`` for
    consistency.

    Parameters
    ----------
    positions : (N, 3)
        Position trajectory in mm.
    timestamps : (N,)
        Strictly increasing timestamps in seconds.

    Returns
    -------
    (N, 3) float64
        Linear velocity in mm/s.
    """
    positions = np.asarray(positions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    n = len(positions)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")
    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must have shape (N, 3), got {positions.shape}"
        )
    _check_strictly_increasing(timestamps)

    velocity = np.empty_like(positions)

    # Forward difference: frame 0
    velocity[0] = (positions[1] - positions[0]) / (timestamps[1] - timestamps[0])

    # Central differences: frames 1..n-2
    if n > 2:
        dt = timestamps[2:] - timestamps[:-2]
        velocity[1:-1] = (positions[2:] - positions[:-2]) / dt[:, np.newaxis]

    # Backward difference: frame n-1
    velocity[-1] = (positions[-1] - positions[-2]) / (timestamps[-1] - timestamps[-2])

    return velocity


def compute_linear_acceleration(
    velocity: NDArray[float64],
    timestamps: NDArray[float64],
) -> NDArray[float64]:
    """Compute linear acceleration from velocity via finite differences.

    Same finite-difference scheme as ``compute_linear_velocity``.
    Typically called with the output of ``compute_linear_velocity``.

    Parameters
    ----------
    velocity : (N, 3)
        Linear velocity in mm/s.
    timestamps : (N,)
        Strictly increasing timestamps in seconds.

    Returns
    -------
    (N, 3) float64
        Linear acceleration in mm/s².
    """
    return compute_linear_velocity(velocity, timestamps)


def compute_angular_acceleration(
    angular_velocity_global: NDArray[float64],
    timestamps: NDArray[float64],
    quaternions: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute angular acceleration from angular velocity.

    Finite differences on angular velocity, then transform to local frame
    using the current orientation at each frame.

    Parameters
    ----------
    angular_velocity_global : (N, 3)
        Angular velocity in the world frame (rad/s).
    timestamps : (N,)
        Strictly increasing timestamps in seconds.
    quaternions : (N, 4)
        Orientation at each frame (used for local-frame transform).

    Returns
    -------
    alpha_global : (N, 3)
        Angular acceleration in the world frame (rad/s²).
    alpha_local : (N, 3)
        Angular acceleration in the body frame (rad/s²), computed as
        Rᵀ @ alpha_global.
    """
    n = len(angular_velocity_global)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    # First derivative of angular velocity (same scheme)
    alpha_global = compute_linear_velocity(
        angular_velocity_global, timestamps
    )

    # Transform to body frame
    R = quaternion_to_rotation_matrix(quaternions)
    alpha_local = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), alpha_global)

    return alpha_global, alpha_local


def compute_keypoint_world_positions(
    local_positions: NDArray[float64],
    quaternions: NDArray[float64],
    origin_positions: NDArray[float64],
) -> NDArray[float64]:
    """Project local keypoint positions to world frame.

    For each frame, rotates the keypoint positions (in the body's local
    frame) by the body's orientation and adds the body's world position.

    Parameters
    ----------
    local_positions : (M, 3)
        Keypoint positions in the body-local frame (e.g. from
        ``SegmentReferenceGeometry`` at T-pose).
    quaternions : (N, 4)
        Body orientation at each frame.
    origin_positions : (N, 3)
        Body origin position at each frame (world frame, mm).

    Returns
    -------
    (N, M, 3) float64
        Keypoint world positions at each frame.
    """
    local_positions = np.asarray(local_positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    origin_positions = np.asarray(origin_positions, dtype=np.float64)

    if local_positions.ndim != 2 or local_positions.shape[1] != 3:
        raise ValueError(
            f"local_positions must have shape (M, 3), got {local_positions.shape}"
        )

    rotated = rotate_vectors_batch(quaternions, local_positions)  # (N, M, 3)
    return rotated + origin_positions[:, np.newaxis, :]


# ═══════════════════════════════════════════════════════════════════════
# Batch model (posthoc analysis)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RigidBodyKinematics:
    """Kinematics for a single rigid body over a full trajectory.

    Stores the raw pose arrays (position + quaternion) and timestamps;
    all derived quantities — velocity, acceleration, angular velocity,
    angular acceleration, Euler angles, keypoint world positions — are
    computed lazily via ``cached_property`` on first access and then
    cached for subsequent calls.

    This is a **batch/posthoc** model. The realtime hot loop should use
    the module-level vectorized functions directly to avoid constructing
    one of these per bone per frame.

    Parameters
    ----------
    name : str
        Identifier for this body (e.g. ``"left_upper_arm"``).
    timestamps : (N,) float64
        Timestamps in seconds, strictly increasing.
    position_xyz : (N, 3) float64
        World-frame position of the body origin at each frame (mm).
    quaternions_wxyz : (N, 4) float64
        Orientation at each frame as [w, x, y, z].
    reference_geometry : SegmentReferenceGeometry or None
        Optional T-pose reference geometry. When provided, enables
        ``keypoint_world_positions`` (propagating the reference
        keypoints to world frame at each frame).
    """

    name: str
    timestamps: NDArray[float64]
    position_xyz: NDArray[float64]
    quaternions_wxyz: NDArray[float64]
    reference_geometry: "SegmentReferenceGeometry | None" = None

    def __post_init__(self) -> None:
        """Validate shapes and normalize quaternions on construction."""
        n = len(self.timestamps)

        if self.position_xyz.shape != (n, 3):
            raise ValueError(
                f"position_xyz must have shape ({n}, 3), "
                f"got {self.position_xyz.shape}"
            )
        if self.quaternions_wxyz.shape != (n, 4):
            raise ValueError(
                f"quaternions_wxyz must have shape ({n}, 4), "
                f"got {self.quaternions_wxyz.shape}"
            )

        if n >= 2:
            _check_strictly_increasing(self.timestamps)

        # Normalize quaternions once at construction time
        self.quaternions_wxyz = normalize_quaternion_array(
            self.quaternions_wxyz
        )

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return len(self.timestamps)

    # ── Derived kinematics (lazy) ─────────────────────────────────

    @cached_property
    def linear_velocity(self) -> NDArray[float64]:
        """(N, 3) Linear velocity in mm/s."""
        return compute_linear_velocity(self.position_xyz, self.timestamps)

    @cached_property
    def linear_acceleration(self) -> NDArray[float64]:
        """(N, 3) Linear acceleration in mm/s²."""
        return compute_linear_acceleration(
            self.linear_velocity, self.timestamps
        )

    @cached_property
    def angular_velocity_global(self) -> NDArray[float64]:
        """(N, 3) Angular velocity in world frame (rad/s)."""
        g, _ = _compute_angular_velocity(
            self.quaternions_wxyz, self.timestamps
        )
        return g

    @cached_property
    def angular_velocity_local(self) -> NDArray[float64]:
        """(N, 3) Angular velocity in body frame (rad/s)."""
        _, l = _compute_angular_velocity(
            self.quaternions_wxyz, self.timestamps
        )
        return l

    @cached_property
    def angular_acceleration_global(self) -> NDArray[float64]:
        """(N, 3) Angular acceleration in world frame (rad/s²)."""
        g, _ = compute_angular_acceleration(
            self.angular_velocity_global,
            self.timestamps,
            self.quaternions_wxyz,
        )
        return g

    @cached_property
    def angular_acceleration_local(self) -> NDArray[float64]:
        """(N, 3) Angular acceleration in body frame (rad/s²)."""
        _, l = compute_angular_acceleration(
            self.angular_velocity_global,
            self.timestamps,
            self.quaternions_wxyz,
        )
        return l

    @cached_property
    def euler_angles(self) -> NDArray[float64]:
        """(N, 3) Euler angles [roll, pitch, yaw] in radians, ZYX intrinsic."""
        return quaternion_to_euler(self.quaternions_wxyz)

    @cached_property
    def keypoint_world_positions(self) -> NDArray[float64] | None:
        """(N, M, 3) World-frame keypoint positions, or None if no reference geometry."""
        if self.reference_geometry is None:
            return None
        local = self.reference_geometry.keypoint_local_positions_array
        return compute_keypoint_world_positions(
            local, self.quaternions_wxyz, self.position_xyz
        )

    # ── Accessors: per-frame ─────────────────────────────────────

    def get_quaternion(self, frame: int) -> RotationQuaternion:
        """Return the ``RotationQuaternion`` at frame index *frame*."""
        if not 0 <= frame < self.n_frames:
            raise IndexError(
                f"frame {frame} out of range [0, {self.n_frames})"
            )
        q = self.quaternions_wxyz[frame]
        return RotationQuaternion(
            w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])
        )

    def get_pose_at_frame(
        self, frame: int
    ) -> tuple[NDArray[float64], RotationQuaternion]:
        """Return ``(position, quaternion)`` at frame index *frame*."""
        if not 0 <= frame < self.n_frames:
            raise IndexError(
                f"frame {frame} out of range [0, {self.n_frames})"
            )
        return self.position_xyz[frame].copy(), self.get_quaternion(frame)

    # ── Component accessors ──────────────────────────────────────

    @cached_property
    def speed(self) -> NDArray[float64]:
        """(N,) Scalar speed (magnitude of linear velocity) in mm/s."""
        return np.linalg.norm(self.linear_velocity, axis=1)

    @cached_property
    def angular_speed_global(self) -> NDArray[float64]:
        """(N,) Scalar angular speed in world frame (rad/s)."""
        return np.linalg.norm(self.angular_velocity_global, axis=1)

    @cached_property
    def angular_speed_local(self) -> NDArray[float64]:
        """(N,) Scalar angular speed in body frame (rad/s)."""
        return np.linalg.norm(self.angular_velocity_local, axis=1)

    # ── Resampling ───────────────────────────────────────────────

    def resample(
        self, target_timestamps: NDArray[float64]
    ) -> "RigidBodyKinematics":
        """Resample to new timestamps.

        Positions are linearly interpolated per axis. Orientations are
        SLERP-interpolated.

        Parameters
        ----------
        target_timestamps : (M,)
            Target timestamps in seconds, monotonically increasing.

        Returns
        -------
        RigidBodyKinematics
            New instance resampled to *target_timestamps*.
        """
        target = np.asarray(target_timestamps, dtype=np.float64)
        if target.ndim != 1:
            raise ValueError(
                f"target_timestamps must be 1D, got shape {target.shape}"
            )
        if len(target) > 1 and not np.all(np.diff(target) >= 0):
            raise ValueError(
                "target_timestamps must be monotonically increasing"
            )

        # Linear interpolation for position (per axis)
        resampled_position = np.column_stack([
            np.interp(target, self.timestamps, self.position_xyz[:, i])
            for i in range(3)
        ])

        # SLERP for orientation
        resampled_quaternions = slerp_resample(
            self.quaternions_wxyz, self.timestamps, target
        )

        return RigidBodyKinematics(
            name=self.name,
            timestamps=target,
            position_xyz=resampled_position,
            quaternions_wxyz=resampled_quaternions,
            reference_geometry=self.reference_geometry,
        )

    def shift_timestamps(self, offset: float) -> "RigidBodyKinematics":
        """Return a new instance with timestamps shifted by *offset* seconds."""
        return RigidBodyKinematics(
            name=self.name,
            timestamps=self.timestamps + offset,
            position_xyz=self.position_xyz.copy(),
            quaternions_wxyz=self.quaternions_wxyz.copy(),
            reference_geometry=self.reference_geometry,
        )

    # ── Factory ──────────────────────────────────────────────────

    @classmethod
    def from_pose_arrays(
        cls,
        name: str,
        timestamps: NDArray[float64],
        position_xyz: NDArray[float64],
        quaternions_wxyz: NDArray[float64],
        reference_geometry: "SegmentReferenceGeometry | None" = None,
    ) -> "RigidBodyKinematics":
        """Construct from raw pose arrays (auto-normalizes quaternions).

        Parameters
        ----------
        name : str
        timestamps : (N,)
        position_xyz : (N, 3)
        quaternions_wxyz : (N, 4)
            May be un-normalized; normalized on construction.
        reference_geometry : SegmentReferenceGeometry or None
        """
        return cls(
            name=name,
            timestamps=np.asarray(timestamps, dtype=np.float64),
            position_xyz=np.asarray(position_xyz, dtype=np.float64),
            quaternions_wxyz=np.asarray(quaternions_wxyz, dtype=np.float64),
            reference_geometry=reference_geometry,
        )


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _check_strictly_increasing(timestamps: NDArray[float64]) -> None:
    """Raise ValueError if timestamps are not strictly increasing."""
    if len(timestamps) < 2:
        return
    dt = np.diff(timestamps)
    bad = np.where(dt <= 1e-10)[0]
    if len(bad) > 0:
        raise ValueError(
            f"Timestamps must be strictly increasing. "
            f"Bad frame {bad[0]} -> {bad[0] + 1}: dt = {dt[bad[0]]:.2e} s"
        )
