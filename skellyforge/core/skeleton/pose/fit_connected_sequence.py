"""Jointly fit a short posthoc window with explicit pose and velocity priors.

No anatomical limits, outlier rejection, or long-recording window stitching are
implied. Missing observations stay missing.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    ConnectedPoseFit,
    LandmarkTarget,
    RootPoseTolerances,
    _prepare_connected_pose,
)
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True)
class ConnectedFrameInput:
    time_seconds: float
    targets: Mapping[str, LandmarkTarget]
    relative_orientations: Mapping[str, RotationQuaternion]
    root_orientation: RotationQuaternion
    root_origin: Point
    pose_prior_orientations: Mapping[str, RotationQuaternion] | None = None
    """Preferred joint pose, independent of initialization; None uses initial pose."""


@dataclass(frozen=True)
class MotionTolerances:
    """Velocity penalty scales, not limits or measured uncertainty."""

    joint_radians_per_second: Mapping[str, float]
    root_radians_per_second: float
    root_units_per_second: float

    def __post_init__(self):
        values = (
            *self.joint_radians_per_second.values(),
            self.root_radians_per_second,
            self.root_units_per_second,
        )
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Motion tolerances must be finite and positive")


@dataclass(frozen=True)
class ConnectedSequenceFit:
    frames: tuple[ConnectedPoseFit, ...]
    initial_cost: float
    final_cost: float
    converged: bool
    termination: str
    evaluations: int
    target_counts: tuple[int, ...]
    """Zero means a frame is supported only by priors and neighboring frames."""


def fit_connected_sequence(
    *,
    skeleton: SkeletonDefinition,
    fit: ModelScaleFit,
    segment_names: frozenset[str],
    frames: Sequence[ConnectedFrameInput],
    rotation_tolerances_radians: Mapping[str, float],
    root_tolerances: RootPoseTolerances,
    motion_tolerances: MotionTolerances,
    max_evaluations: int = 100,
) -> ConnectedSequenceFit:
    """Fit every frame in the supplied window simultaneously, including its roots.

    Integrate observation/pose costs with trapezoidal timestamp weights. Integrate
    squared angular/linear velocity over each interval: residual = displacement /
    (velocity_scale * sqrt(dt)). Angular displacement is the principal SO(3) log,
    not a quaternion-component difference. All scales are explicit caller inputs.

    Explicit pose_prior_orientations define joint priors; otherwise initial poses
    also define them. Root priors remain relative to the supplied root. No endpoint
    is pinned. Neighbor terms
    span the supplied intervals, including gaps; callers must split discontinuous
    takes. At least two frames and one actual observation are required. A missing
    frame is inferred, not measured; target_counts reports it. Per-frame costs in
    returned frames exclude quadrature weights and temporal terms. Global costs
    include both. `evaluations` counts SciPy objective evaluations, not iterations.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    times = np.asarray([f.time_seconds for f in frames], dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(
            "At least two finite, strictly increasing timestamps are required"
        )
    if not any(f.targets for f in frames):
        raise ValueError("Sequence requires at least one observation")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    movable = tuple(sorted(rotation_tolerances_radians))
    if set(movable) != set(motion_tolerances.joint_radians_per_second):
        raise ValueError("Motion tolerances must cover exactly the movable joints")
    dt = np.diff(times)
    weights = np.zeros(len(times))
    weights[:-1] += dt / 2
    weights[1:] += dt / 2
    prepared = [
        _prepare_connected_pose(
            skeleton=skeleton,
            fit=fit,
            segment_names=segment_names,
            segment_relative_orientations=f.relative_orientations,
            root_world_orientation=f.root_orientation,
            root_origin=f.root_origin,
            targets=f.targets,
            rotation_tolerances_radians=rotation_tolerances_radians,
            root_tolerances=root_tolerances,
            pose_prior_orientations=f.pose_prior_orientations,
            allow_missing_targets=True,
        )
        for f in frames
    ]
    size = prepared[0][1]
    roots = segment_names - {j.child.name for j in skeleton.joints.values()}
    (root_name,) = roots
    # Cache only the latest evaluation of each frame. Neighbor differences need
    # no new FK when SciPy perturbs variables belonging to other frames.
    cache = [None] * len(frames)

    def evaluate_frame(i, delta):
        previous = cache[i]
        if previous is None or not np.array_equal(previous[0], delta):
            cache[i] = (delta.copy(), prepared[i][0](delta))
        return cache[i][1]

    def evaluate(flat):
        states = flat.reshape(len(frames), size)
        evaluated = [evaluate_frame(i, delta) for i, delta in enumerate(states)]
        residuals = [
            value[0] * np.sqrt(weights[i]) for i, value in enumerate(evaluated)
        ]
        for i, interval in enumerate(dt):
            world_a, origins_a, _, local_a = evaluated[i][1]
            world_b, origins_b, _, local_b = evaluated[i + 1][1]
            for name in movable:
                difference = (
                    local_a[name].conjugate() * local_b[name]
                ).to_rotation_vector()
                residuals.append(
                    difference
                    / (
                        motion_tolerances.joint_radians_per_second[name]
                        * np.sqrt(interval)
                    )
                )
            difference = (
                world_a[root_name].conjugate() * world_b[root_name]
            ).to_rotation_vector()
            residuals.append(
                difference
                / (motion_tolerances.root_radians_per_second * np.sqrt(interval))
            )
            residuals.append(
                (origins_b[root_name].array - origins_a[root_name].array)
                / (motion_tolerances.root_units_per_second * np.sqrt(interval))
            )
        return np.concatenate(residuals)

    initial = np.zeros(len(frames) * size)
    initial_residual = evaluate(initial)
    # Each observation/pose block touches one frame; each velocity block two.
    sparsity = lil_matrix((len(initial_residual), len(initial)), dtype=int)
    row = 0
    for i, f in enumerate(frames):
        count = 3 * len(f.targets) + size
        sparsity[row : row + count, i * size : (i + 1) * size] = 1
        row += count
    for i in range(len(frames) - 1):
        sparsity[row : row + size, i * size : (i + 2) * size] = 1
        row += size
    optimized = least_squares(
        evaluate,
        initial,
        method="trf",
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        max_nfev=max_evaluations,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )
    output = []
    for i, delta in enumerate(optimized.x.reshape(len(frames), size)):
        residual, (world, origins, landmarks, local) = evaluate_frame(i, delta)
        errors = {
            n: float(np.linalg.norm(landmarks[n].array - t.position.array))
            for n, t in frames[i].targets.items()
        }
        initial_frame_residual = prepared[i][0](np.zeros(size))[0]
        output.append(
            ConnectedPoseFit(
                world,
                origins,
                landmarks,
                local,
                errors,
                float(initial_frame_residual @ initial_frame_residual),
                float(residual @ residual),
                None,
                str(optimized.message),
                bool(errors)
                and all(
                    error <= frames[i].targets[n].tolerance
                    for n, error in errors.items()
                ),
            )
        )
    return ConnectedSequenceFit(
        tuple(output),
        float(initial_residual @ initial_residual),
        float(optimized.fun @ optimized.fun),
        bool(optimized.success),
        str(optimized.message),
        optimized.nfev,
        tuple(len(f.targets) for f in frames),
    )
