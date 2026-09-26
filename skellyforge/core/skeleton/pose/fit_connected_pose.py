"""Fit connected, fixed-dimension poses to multiple landmark targets.

Regularized nonlinear least squares over parent-frame rotation vectors. A finite
difference Jacobian and damped least-squares steps need only NumPy. Every trial
pose uses the existing forward kinematics, so lengths and connections are hard
constraints. Target/rotation tolerances define the model preference; numerical
damping only controls the optimizer. This is a local solve, not global IK or an
anatomical joint-limit model.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import (
    synthesize_fitted_pose,
    synthesize_pose,
)
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True)
class LandmarkTarget:
    position: Point
    tolerance: float
    """Position residual scale in the fit's units; a modeling weight, not measured uncertainty."""


@dataclass(frozen=True)
class RootPoseTolerances:
    """Explicit priors toward the supplied root pose; translation uses fit units."""

    translation: float
    rotation_radians: float

    def __post_init__(self):
        if any(
            not np.isfinite(v) or v <= 0
            for v in (self.translation, self.rotation_radians)
        ):
            raise ValueError("Root tolerances must be finite and positive")


@dataclass(frozen=True)
class ConnectedPoseFit:
    world_orientations: dict[str, RotationQuaternion]
    world_origins: dict[str, Point]
    landmarks: dict[str, Point]
    relative_orientations: dict[str, RotationQuaternion]
    target_errors: dict[str, float]
    initial_cost: float
    final_cost: float
    iterations: int | None
    """None when the optimizer reports evaluations instead of iteration count."""
    termination: str
    targets_within_tolerance: bool
    """Independent of optimizer termination: a converged compromise can miss targets."""


def _prepare_connected_pose(
    *,
    skeleton: SkeletonDefinition,
    fit: ModelScaleFit,
    segment_relative_orientations: Mapping[str, RotationQuaternion],
    root_world_orientation: RotationQuaternion,
    root_origin: Point,
    segment_names: frozenset[str],
    targets: Mapping[str, LandmarkTarget],
    rotation_tolerances_radians: Mapping[str, float],
    root_tolerances: RootPoseTolerances | None = None,
    pose_prior_orientations: Mapping[str, RotationQuaternion] | None = None,
    allow_missing_targets: bool = False,
):
    """Fit selected rotations and, optionally, the root pose jointly.

    Root remains fixed unless root_tolerances is supplied. Root increments are
    expressed in the initial root frame; translation is scaled by fitted_scale
    for dimensionless numerical coordinates. Root priors penalize displacement
    and rotation from the initial pose, not from world zero.

    Keys in rotation_tolerances_radians are exactly the movable non-root segments.
    All other selected local rotations remain unchanged. For each movable segment,
    q = exp(delta) * q_reference, with delta expressed in its parent's coordinates.
    Minimize sum(||position_error / target_tolerance||^2) plus
    sum(||delta / rotation_tolerance||^2). Tolerances must be explicit and positive.

    No temporal smoothing, anatomical limits, missing-target inference, or source
    mutation. Nonfinite targets fail. Callers may omit unavailable targets, but at
    least one is required. Results report iteration exhaustion/stalling explicitly.
    """
    if (not targets and not allow_missing_targets) or (
        not rotation_tolerances_radians and root_tolerances is None
    ):
        raise ValueError("At least one target and movable segment or root are required")
    movable = tuple(sorted(rotation_tolerances_radians))
    if pose_prior_orientations is not None and not set(movable).issubset(
        pose_prior_orientations
    ):
        raise ValueError("Pose priors must cover every movable joint")
    if not set(movable).issubset(segment_relative_orientations):
        raise ValueError("Movable segments must be selected non-root segments")
    angular_scales = np.repeat([rotation_tolerances_radians[n] for n in movable], 3)
    if not np.isfinite(angular_scales).all() or np.any(angular_scales <= 0):
        raise ValueError("Rotation tolerances must be finite and positive")
    for name, target in targets.items():
        if (
            name not in skeleton.landmarks
            or skeleton.landmarks[name].segment not in segment_names
        ):
            raise ValueError(f"Target {name!r} must belong to a selected segment")
        if (
            target.position.array.shape != (3,)
            or not np.isfinite(target.position.array).all()
        ):
            raise ValueError(f"Target {name!r} must be one finite point")
        if not np.isfinite(target.tolerance) or target.tolerance <= 0:
            raise ValueError("Position tolerances must be finite and positive")

    # Validate fit, selection, root and complete rotations through the public FK boundary.
    synthesize_fitted_pose(
        skeleton=skeleton,
        fit=fit,
        segment_relative_orientations=segment_relative_orientations,
        root_world_orientation=root_world_orientation,
        root_origin=root_origin,
        segment_names=segment_names,
    )
    reference = dict(segment_relative_orientations)
    joint_names = {
        j.child.name: j.name
        for j in skeleton.joints.values()
        if j.child.name in segment_names
    }
    target_names = tuple(sorted(targets))
    angular_count = 3 * len(movable)
    prior_scales = angular_scales
    if root_tolerances is not None:
        prior_scales = np.concatenate(
            (
                angular_scales,
                np.repeat(root_tolerances.rotation_radians, 3),
                np.repeat(root_tolerances.translation / fit.fitted_scale, 3),
            )
        )

    def evaluate(delta):
        local = dict(reference)
        for i, name in enumerate(movable):
            local[name] = (
                RotationQuaternion.from_rotation_vector(
                    rotation_vector=delta[3 * i : 3 * i + 3]
                )
                * reference[name]
            )
        root_q, root_p = root_world_orientation, root_origin
        if root_tolerances is not None:
            root_q = root_world_orientation * RotationQuaternion.from_rotation_vector(
                rotation_vector=delta[angular_count : angular_count + 3]
            )
            root_p = Point.from_array(
                values=root_origin.array
                + root_world_orientation.rotate_vector(
                    vector=delta[angular_count + 3 :] * fit.fitted_scale
                )
            )
        world, origins, landmarks = synthesize_pose(
            skeleton=skeleton,
            joint_relative_orientations={joint_names[n]: q for n, q in local.items()},
            root_world_orientation=root_q,
            root_origin=root_p,
            segment_scales=fit.segment_scales,
            segment_names=segment_names,
        )
        positional = (
            np.concatenate(
                [
                    (landmarks[n].array - targets[n].position.array)
                    / targets[n].tolerance
                    for n in target_names
                ]
            )
            if target_names
            else np.empty(0)
        )
        prior = delta / prior_scales
        if pose_prior_orientations is not None:
            for i, name in enumerate(movable):
                prior[3 * i : 3 * i + 3] = (
                    pose_prior_orientations[name].conjugate() * local[name]
                ).to_rotation_vector() / rotation_tolerances_radians[name]
        residual = np.concatenate((positional, prior))
        return residual, (world, origins, landmarks, local)

    rotation_count = angular_count + (3 if root_tolerances is not None else 0)
    return evaluate, len(prior_scales), rotation_count


def fit_connected_pose(
    *,
    skeleton: SkeletonDefinition,
    fit: ModelScaleFit,
    segment_relative_orientations: Mapping[str, RotationQuaternion],
    root_world_orientation: RotationQuaternion,
    root_origin: Point,
    segment_names: frozenset[str],
    targets: Mapping[str, LandmarkTarget],
    rotation_tolerances_radians: Mapping[str, float],
    root_tolerances: RootPoseTolerances | None = None,
    pose_prior_orientations: Mapping[str, RotationQuaternion] | None = None,
    max_iterations: int = 60,
    gradient_tolerance: float = 1e-6,
) -> ConnectedPoseFit:
    """Fit connected geometry with explicit pose priors and optional root motion.

    Joint increments are in parent coordinates; root increments are in the
    initial root frame. All tolerances must be positive. Missing observations
    may be omitted, but this single-frame fitter requires at least one target.
    No temporal smoothing or anatomical limits are implied by this local solve.
    pose_prior_orientations separates the preferred joint pose from the starting
    guess. If omitted, the starting guess also supplies that preference.
    """
    if (
        max_iterations < 1
        or not np.isfinite(gradient_tolerance)
        or gradient_tolerance <= 0
    ):
        raise ValueError("Iterations and gradient tolerance must be positive")
    evaluate, size, rotation_count = _prepare_connected_pose(
        skeleton=skeleton,
        fit=fit,
        segment_relative_orientations=segment_relative_orientations,
        root_world_orientation=root_world_orientation,
        root_origin=root_origin,
        segment_names=segment_names,
        targets=targets,
        rotation_tolerances_radians=rotation_tolerances_radians,
        root_tolerances=root_tolerances,
        pose_prior_orientations=pose_prior_orientations,
    )
    target_names = tuple(sorted(targets))

    delta = np.zeros(size)
    residual, geometry = evaluate(delta)
    initial_cost = cost = float(residual @ residual)
    if not np.isfinite(cost):
        raise ValueError(
            "Target magnitudes and tolerances must produce a finite objective"
        )
    damping = 1e-3
    difference_step = np.cbrt(np.finfo(float).eps)
    termination = "iteration_limit"
    for iteration in range(1, max_iterations + 1):
        jacobian = np.empty((len(residual), len(delta)))
        for column in range(len(delta)):
            step = np.zeros_like(delta)
            step[column] = difference_step
            jacobian[:, column] = (
                evaluate(delta + step)[0] - evaluate(delta - step)[0]
            ) / (2 * difference_step)
        gradient = jacobian.T @ residual
        if np.linalg.norm(gradient, ord=np.inf) <= gradient_tolerance:
            termination = "stationary"
            break
        diagonal = np.maximum(np.sum(jacobian * jacobian, axis=0), 1.0)
        accepted = False
        for _ in range(16):
            augmented = np.vstack((jacobian, np.diag(np.sqrt(damping * diagonal))))
            step = np.linalg.lstsq(
                augmented, np.concatenate((-residual, np.zeros(len(delta)))), rcond=None
            )[0]
            candidate = delta + step
            # Stay in the principal rotation-vector neighborhood of the reference.
            if np.any(
                np.linalg.norm(candidate[:rotation_count].reshape(-1, 3), axis=1)
                >= np.pi
            ):
                damping *= 10
                continue
            trial_residual, trial_geometry = evaluate(candidate)
            trial_cost = float(trial_residual @ trial_residual)
            if np.isfinite(trial_cost) and trial_cost < cost:
                delta, residual, geometry, cost = (
                    candidate,
                    trial_residual,
                    trial_geometry,
                    trial_cost,
                )
                damping = max(damping / 3, 1e-12)
                accepted = True
                break
            damping *= 10
        if not accepted:
            termination = "stalled"
            break
    world, origins, landmarks, local = geometry
    errors = {
        n: float(np.linalg.norm(landmarks[n].array - targets[n].position.array))
        for n in target_names
    }
    return ConnectedPoseFit(
        world_orientations=world,
        world_origins=origins,
        landmarks=landmarks,
        relative_orientations=local,
        target_errors=errors,
        initial_cost=initial_cost,
        final_cost=cost,
        iterations=iteration,
        termination=termination,
        targets_within_tolerance=all(
            errors[n] <= targets[n].tolerance for n in target_names
        ),
    )
