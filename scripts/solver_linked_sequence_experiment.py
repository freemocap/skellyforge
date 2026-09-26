"""A linked Ceres sequence with a bounded interval of missing child observations."""

import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts.solver_linkage_experiment import linkage_inputs


def linked_sequence_experiment(*, noise, gap):
    local, attachments, times, records = linkage_inputs(noise=noise, seed=7)
    missing = set(range(20 - gap // 2, 20 + gap // 2 + 1)) if gap else set()
    parent_observed = [record[5][0].tolist() for record in records]
    child_observed = [
        record[5][1].tolist() if i not in missing else []
        for i, record in enumerate(records)
    ]
    result = _native.fit_linked_sequence(
        local_a=local.tolist(),
        observed_a=parent_observed,
        attachment_a=attachments[0].tolist(),
        local_b=local.tolist(),
        observed_b=child_observed,
        attachment_b=attachments[1].tolist(),
        times=times.tolist(),
        position_scale=10.0,
        linear_acceleration_scale=3000.0,
        angular_acceleration_scale=20.0,
    )
    methods = {name: dict(frames=[]) for name in ["per_frame", "temporal"]}
    for i, (
        time,
        known_joint,
        rotations,
        translations,
        truth,
        observations,
    ) in enumerate(records):
        if i in missing:
            baseline = _native.fit_rigid(
                local=local.tolist(),
                observed=parent_observed[i],
                quaternion=[1.0, 0.0, 0.0, 0.0],
                translation=[0.0, 0.0, 0.0],
            )
            baseline_q = [baseline.quaternion, None]
            baseline_t = [baseline.translation, None]
            baseline_joint = (
                Rotation.from_quat(baseline.quaternion, scalar_first=True).apply(
                    attachments[0]
                )
                + baseline.translation
            )
        else:
            baseline = _native.fit_linked_rigid(
                local_a=local.tolist(),
                observed_a=parent_observed[i],
                attachment_a=attachments[0].tolist(),
                local_b=local.tolist(),
                observed_b=child_observed[i],
                attachment_b=attachments[1].tolist(),
            )
            baseline_q, baseline_t, baseline_joint = (
                baseline.quaternions,
                baseline.translations,
                baseline.joint,
            )
        for name, method in methods.items():
            sequence = name == "temporal"
            qs = result.quaternions[i] if sequence else baseline_q
            ts = result.translations[i] if sequence else baseline_t
            bodies = []
            for b in range(2):
                available = qs[b] is not None
                observed = (
                    observations[b].tolist()
                    if b == 0 or i not in missing
                    else [None] * 8
                )
                fitted = (
                    Rotation.from_quat(qs[b], scalar_first=True).apply(local) + ts[b]
                    if available
                    else None
                )
                initial_q = result.initial_quaternions[i][b] if sequence else None
                initial_t = result.initial_translations[i][b] if sequence else None
                bodies.append(
                    dict(
                        available=available,
                        local=local.tolist(),
                        truth=truth[b].tolist(),
                        observed=observed,
                        fitted=fitted.tolist() if available else [None] * 8,
                        quaternion=qs[b],
                        translation=ts[b],
                        initial_quaternion=initial_q,
                        initial_translation=initial_t,
                        initial=(
                            (
                                Rotation.from_quat(initial_q, scalar_first=True).apply(
                                    local
                                )
                                + initial_t
                            ).tolist()
                            if sequence
                            else None
                        ),
                        reference_quaternion=rotations[b]
                        .as_quat(scalar_first=True)
                        .tolist(),
                        reference_translation=translations[b].tolist(),
                        residuals=[
                            (
                                float(np.linalg.norm(fitted[k] - observations[b, k]))
                                if available and observed[k] is not None
                                else None
                            )
                            for k in range(8)
                        ],
                        truth_rms=(
                            float(
                                np.sqrt(
                                    np.mean(np.sum((fitted - truth[b]) ** 2, axis=1))
                                )
                            )
                            if available
                            else None
                        ),
                    )
                )
            diagnostics = {"Child landmark residual blocks": 0 if i in missing else 8}
            if sequence or i not in missing:
                attachment_positions = [
                    Rotation.from_quat(q, scalar_first=True).apply(a) + t
                    for q, a, t in zip(qs, attachments, ts)
                ]
                diagnostics["Attachment separation (mm)"] = float(
                    np.linalg.norm(attachment_positions[0] - attachment_positions[1])
                )
                diagnostics["Child landmarks error vs known (mm)"] = bodies[1][
                    "truth_rms"
                ]
            method["frames"].append(
                dict(
                    time=float(time),
                    bodies=bodies,
                    joint=(
                        result.joints[i]
                        if sequence
                        else np.asarray(baseline_joint).tolist()
                    ),
                    costs=result.costs if sequence else baseline.costs,
                    converged=result.converged if sequence else baseline.converged,
                    report=result.report if sequence else baseline.report,
                    seconds=result.seconds if sequence else baseline.seconds,
                    diagnostics=diagnostics,
                    observability=(
                        (
                            "Child has no landmark residuals here. Its pose is inferred through temporal residual blocks, not directly observed."
                            if sequence
                            else "No child pose is available: this frame has no child observations and no temporal residuals."
                        )
                        if i in missing
                        else ""
                    ),
                    problem=dict(
                        connected=sequence or i not in missing,
                        temporal=sequence,
                        acceleration=sequence,
                    ),
                )
            )
    for name, method in methods.items():
        sequence = name == "temporal"
        method["problem"] = dict(
            connected=True, temporal=sequence, acceleration=sequence
        )
        method["settings"] = dict(
            position_scale_mm=10.0 if sequence else None,
            linear_motion_scale=3000.0 if sequence else None,
            angular_motion_scale=20.0 if sequence else None,
            scale_units=["mm/s^2", "rad/s^2"],
            temporal="acceleration residual blocks" if sequence else "none",
            loss="squared",
            initialization=(
                "Connected observed-frame fits; parent fits in gaps; child quaternion Slerp between surrounding observed-frame fits. Initialization only; all blocks optimized."
                if sequence
                else "Independent per-frame fits; no child estimate in gaps."
            ),
        )
        method["summary"] = {
            "Child frames without a fitted pose": sum(
                f["bodies"][1]["quaternion"] is None for f in method["frames"]
            )
        }
        errors = [
            f["bodies"][1]["truth_rms"]
            for j, f in enumerate(method["frames"])
            if j in missing and f["bodies"][1]["truth_rms"] is not None
        ]
        if errors:
            method["summary"]["Child gap landmark RMS vs known (mm)"] = float(
                np.sqrt(np.mean(np.square(errors)))
            )
        if sequence:
            method["summary"].update(
                {
                    "Ceres parameter blocks": result.parameter_blocks,
                    "Ceres residual blocks": result.residual_blocks,
                }
            )
            method["settings"]["costs_by_family"] = dict(
                landmark=result.landmark_cost,
                joint_acceleration=result.joint_acceleration_cost,
                parent_angular_acceleration=result.parent_acceleration_cost,
                child_angular_acceleration=result.child_acceleration_cost,
            )
        method["objective"] = (
            "Whole-sequence scaled landmark residuals plus joint-position and two quaternion acceleration residual families. Shared-point linkage is exact."
            if sequence
            else "Selected-frame landmark objective. A missing child has no fitted pose."
        )
    return dict(
        parameters=dict(noise=noise, gap=gap),
        times=times.tolist(),
        methods=methods,
        gap_frames=sorted(missing),
        gap_times=(
            [float(times[min(missing)]), float(times[max(missing)])]
            if missing
            else None
        ),
    )
