"""Synthetic sequences and diagnostics; no reference poses enter the Ceres fit."""

import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native


def sequence_experiment(*, noise, missing, outlier, seed, moving):
    local = np.array(list(itertools.product([-100.0, 100.0], repeat=3)))
    times = np.linspace(0, 2, 41)
    phase = 2 * np.pi * times
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    angles = (
        np.deg2rad(45 + 25 * np.sin(phase))
        if moving
        else np.full(len(times), np.pi / 4)
    )
    quaternions = np.column_stack(
        [np.cos(angles / 2), np.sin(angles / 2)[:, None] * axis]
    )
    translations = np.tile([60.0, 30.0, 180.0], (len(times), 1))
    if moving:
        translations[:, 0] += 100 * np.sin(phase)
        translations[:, 2] += 30 * np.sin(phase / 2)
    truth = np.array(
        [
            Rotation.from_quat(q, scalar_first=True).apply(local) + t
            for q, t in zip(quaternions, translations)
        ]
    )
    observed = truth + np.random.default_rng(seed).normal(0, noise, truth.shape)
    # A single bad frame demonstrates the influence on neighbors in the sequence fit.
    observed[len(times) // 2, 0, 0] += outlier
    count = 8 - missing
    independent = [
        _native.fit_rigid(
            local=local[:count].tolist(),
            observed=frame[:count].tolist(),
            quaternion=[1.0, 0.0, 0.0, 0.0],
            translation=[0.0, 0.0, 0.0],
        )
        for frame in observed
    ]
    modes = {}
    for mode, linear, angular in [
        ("independent", None, None),
        ("sequence", 300.0, 2.0),
        ("strong", 60.0, 0.4),
        ("acceleration", 3000.0, 20.0),
    ]:
        if mode == "independent":
            qs = [r.quaternion for r in independent]
            ts = [r.translation for r in independent]
            result = None
        else:
            result = _native.fit_rigid_sequence(
                local=local[:count].tolist(),
                observed=observed[:, :count].tolist(),
                times=times.tolist(),
                position_scale=10.0,
                linear_motion_scale=linear,
                angular_motion_scale=angular,
                temporal_model="acceleration" if mode == "acceleration" else "velocity",
            )
            qs = result.quaternions
            ts = result.translations
        fitted = np.array(
            [
                Rotation.from_quat(q, scalar_first=True).apply(local) + t
                for q, t in zip(qs, ts)
            ]
        )
        frames = []
        for i, (q, t) in enumerate(zip(qs, ts)):
            base = independent[i]
            initial_q = base.quaternion if result else [1.0, 0.0, 0.0, 0.0]
            initial_t = base.translation if result else [0.0, 0.0, 0.0]
            frames.append(
                dict(
                    time=float(times[i]),
                    local=local.tolist(),
                    truth=truth[i].tolist(),
                    initial=(
                        Rotation.from_quat(initial_q, scalar_first=True).apply(local)
                        + initial_t
                    ).tolist(),
                    initial_quaternion=initial_q,
                    initial_translation=initial_t,
                    observed=observed[i, :count].tolist(),
                    fitted=fitted[i].tolist(),
                    quaternion=q,
                    translation=t,
                    reference_quaternion=quaternions[i].tolist(),
                    reference_translation=translations[i].tolist(),
                    costs=result.costs if result else base.costs,
                    converged=result.converged if result else base.converged,
                    report=result.report if result else base.report,
                    seconds=result.seconds if result else base.seconds,
                    residuals=np.linalg.norm(
                        fitted[i, :count] - observed[i, :count], axis=1
                    ).tolist(),
                    truth_rms=float(
                        np.sqrt(np.mean(np.sum((fitted[i] - truth[i]) ** 2, axis=1)))
                    ),
                    rotation_error_degrees=float(
                        np.degrees(
                            2 * np.arccos(np.clip(abs(np.dot(q, quaternions[i])), 0, 1))
                        )
                    ),
                )
            )
        modes[mode] = dict(
            frames=frames,
            settings=dict(
                position_scale_mm=10.0 if result else None,
                linear_motion_scale=linear,
                angular_motion_scale=angular,
                scale_units=(
                    ["mm/s^2", "rad/s^2"]
                    if mode == "acceleration"
                    else ["mm/s", "rad/s"]
                ),
                rotation_penalty=(
                    "world angular velocity difference"
                    if mode == "acceleration"
                    else "quaternion chord"
                ),
                loss="squared",
                temporal=(
                    (
                        "integrated squared acceleration preference"
                        if mode == "acceleration"
                        else "integrated squared velocity preference"
                    )
                    if result
                    else "none"
                ),
            ),
            costs_by_family=(
                dict(
                    landmark=result.landmark_cost,
                    translation=result.translation_cost,
                    rotation=result.rotation_cost,
                )
                if result
                else None
            ),
            mean_corner_error_mm=float(np.sqrt(np.mean((fitted - truth) ** 2) * 3)),
            translation_x_range_mm=float(np.ptp(np.array(ts)[:, 0])),
            known_translation_x_range_mm=float(np.ptp(translations[:, 0])),
        )
    return dict(
        noise=noise,
        missing=missing,
        outlier=outlier,
        seed=seed,
        motion="moving" if moving else "stationary",
        modes=modes,
        times=times.tolist(),
        units="mm",
        quaternion_order="wxyz",
        ceres_version=_native.ceres_version,
        outlier_frame=20,
        missing_ids=list(range(count, 8)),
        initialization="independent Ceres fits; no reference poses",
        reference_frequency_hz=1.0 if moving else 0.0,
    )
