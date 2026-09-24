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
class ConnectedPoseFit:
    world_orientations: dict[str, RotationQuaternion]
    world_origins: dict[str, Point]
    landmarks: dict[str, Point]
    relative_orientations: dict[str, RotationQuaternion]
    target_errors: dict[str, float]
    initial_cost: float
    final_cost: float
    iterations: int
    termination: str
    targets_within_tolerance: bool
    """Independent of optimizer termination: a converged compromise can miss targets."""


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
    max_iterations: int = 60,
    gradient_tolerance: float = 1e-6,
) -> ConnectedPoseFit:
    """Fit selected rotations jointly, keeping the root and other local rotations fixed.

    Keys in rotation_tolerances_radians are exactly the movable non-root segments.
    All other selected local rotations remain unchanged. For each movable segment,
    q = exp(delta) * q_reference, with delta expressed in its parent's coordinates.
    Minimize sum(||position_error / target_tolerance||^2) plus
    sum(||delta / rotation_tolerance||^2). Tolerances must be explicit and positive.

    No temporal smoothing, anatomical limits, missing-target inference, or source
    mutation. Nonfinite targets fail. Callers may omit unavailable targets, but at
    least one is required. Results report iteration exhaustion/stalling explicitly.
    """
    if (
        max_iterations < 1
        or not np.isfinite(gradient_tolerance)
        or gradient_tolerance <= 0
    ):
        raise ValueError("Iterations and gradient tolerance must be positive")
    if not targets or not rotation_tolerances_radians:
        raise ValueError("At least one target and movable segment are required")
    movable = tuple(sorted(rotation_tolerances_radians))
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

    def evaluate(delta):
        local = dict(reference)
        for i, name in enumerate(movable):
            local[name] = (
                RotationQuaternion.from_rotation_vector(
                    rotation_vector=delta[3 * i : 3 * i + 3]
                )
                * reference[name]
            )
        world, origins, landmarks = synthesize_pose(
            skeleton=skeleton,
            joint_relative_orientations={joint_names[n]: q for n, q in local.items()},
            root_world_orientation=root_world_orientation,
            root_origin=root_origin,
            segment_scales=fit.segment_scales,
            segment_names=segment_names,
        )
        positional = np.concatenate(
            [
                (landmarks[n].array - targets[n].position.array) / targets[n].tolerance
                for n in target_names
            ]
        )
        residual = np.concatenate((positional, delta / angular_scales))
        return residual, (world, origins, landmarks, local)

    delta = np.zeros(3 * len(movable))
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
            if np.any(np.linalg.norm(candidate.reshape(-1, 3), axis=1) >= np.pi):
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
