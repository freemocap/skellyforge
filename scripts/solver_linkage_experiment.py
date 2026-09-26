"""Two rigid segments sharing one point; independent frames, no temporal prior."""

import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native


def rotation(axis, angle):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return Rotation.from_quat(
        np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)], scalar_first=True
    )


def linkage_inputs(*, noise, seed):
    local = np.array(
        list(itertools.product([-30.0, 30.0], [-30.0, 30.0], [-100.0, 100.0]))
    )
    attachments = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, -100.0]])
    times = np.linspace(0, 2, 41)
    rng = np.random.default_rng(seed)
    records = []
    for time in times:
        phase = 2 * np.pi * time / 2
        known_joint = np.array([40 * np.sin(phase), 20 * np.cos(phase), 240.0])
        rotations = [
            rotation([1, 2, 0], 0.3 * np.sin(phase)),
            rotation([1, 0.2, 0], 0.7 + 0.5 * np.sin(phase)),
        ]
        translations = [
            known_joint - r.apply(a) for r, a in zip(rotations, attachments)
        ]
        truth = np.array([r.apply(local) + t for r, t in zip(rotations, translations)])
        observed = truth + rng.normal(0, noise, truth.shape)
        records.append((time, known_joint, rotations, translations, truth, observed))
    return local, attachments, times, records


def linkage_experiment(*, noise, seed, child_points=8):
    local, attachments, times, records = linkage_inputs(noise=noise, seed=seed)
    counts = [8, child_points]
    methods = {name: dict(frames=[]) for name in ["independent", "connected"]}
    for time, known_joint, rotations, translations, truth, observed in records:
        independent = [
            _native.fit_rigid(
                local=local[: counts[i]].tolist(),
                observed=o[: counts[i]].tolist(),
                quaternion=[1.0, 0.0, 0.0, 0.0],
                translation=[0.0, 0.0, 0.0],
                allow_underconstrained=i == 1,
            )
            for i, o in enumerate(observed)
        ]
        connected = _native.fit_linked_rigid(
            local_a=local.tolist(),
            observed_a=observed[0].tolist(),
            attachment_a=attachments[0].tolist(),
            local_b=local[:child_points].tolist(),
            observed_b=observed[1, :child_points].tolist(),
            attachment_b=attachments[1].tolist(),
        )
        initial_rotations = [
            Rotation.from_quat(f.quaternion, scalar_first=True) for f in independent
        ]
        initial_joint = np.mean(
            [
                r.apply(a) + f.translation
                for r, a, f in zip(initial_rotations, attachments, independent)
            ],
            axis=0,
        )
        for name, method in methods.items():
            qs = (
                connected.quaternions
                if name == "connected"
                else [f.quaternion for f in independent]
            )
            ts = (
                connected.translations
                if name == "connected"
                else [f.translation for f in independent]
            )
            bodies = []
            joints = []
            for i, (q, t) in enumerate(zip(qs, ts)):
                fitted = Rotation.from_quat(q, scalar_first=True).apply(local) + t
                joints.append(
                    Rotation.from_quat(q, scalar_first=True).apply(attachments[i]) + t
                )
                initial_q = (
                    independent[i].quaternion
                    if name == "connected"
                    else [1.0, 0.0, 0.0, 0.0]
                )
                initial_t = (
                    initial_joint - initial_rotations[i].apply(attachments[i])
                    if name == "connected"
                    else np.zeros(3)
                )
                bodies.append(
                    dict(
                        local=local.tolist(),
                        truth=truth[i].tolist(),
                        observed=[
                            p.tolist() if j < counts[i] else None
                            for j, p in enumerate(observed[i])
                        ],
                        fitted=fitted.tolist(),
                        initial=(
                            Rotation.from_quat(initial_q, scalar_first=True).apply(
                                local
                            )
                            + initial_t
                        ).tolist(),
                        initial_quaternion=initial_q,
                        initial_translation=initial_t.tolist(),
                        reference_quaternion=rotations[i]
                        .as_quat(scalar_first=True)
                        .tolist(),
                        reference_translation=translations[i].tolist(),
                        quaternion=q,
                        translation=t,
                        residuals=[
                            float(r) if j < counts[i] else None
                            for j, r in enumerate(
                                np.linalg.norm(fitted - observed[i], axis=1)
                            )
                        ],
                        withheld_error_mm=np.linalg.norm(
                            fitted[counts[i] :] - truth[i, counts[i] :], axis=1
                        ).tolist(),
                        truth_rms=float(
                            np.sqrt(np.mean(np.sum((fitted - truth[i]) ** 2, axis=1)))
                        ),
                    )
                )
            # Independent segment histories have different lengths; do not invent a joint iteration history.
            costs = connected.costs if name == "connected" else []
            method["frames"].append(
                dict(
                    time=float(time),
                    bodies=bodies,
                    costs=costs,
                    converged=(
                        connected.converged
                        if name == "connected"
                        else all(f.converged for f in independent)
                    ),
                    seconds=(
                        connected.seconds
                        if name == "connected"
                        else sum(f.seconds for f in independent)
                    ),
                    report=(
                        connected.report
                        if name == "connected"
                        else "Two independent Ceres solves; separate iteration histories in segment_reports."
                    ),
                    segment_reports=[
                        dict(report=f.report, costs=f.costs) for f in independent
                    ],
                    observability=(
                        "Child roll about attachment-to-observed-landmark axis is unconstrained; fitted roll depends on initialization."
                        if child_points == 1 and name == "connected"
                        else (
                            "Child rotation about the observed landmark line is unconstrained."
                            if child_points == 2 and name == "independent"
                            else (
                                "Child rotation is unconstrained by a single observed landmark."
                                if child_points == 1
                                else "Sufficient non-collinear support for both rotations."
                            )
                        )
                    ),
                    diagnostics={
                        **(
                            {
                                "Withheld child landmarks RMS (mm)": float(
                                    np.sqrt(
                                        np.mean(
                                            np.square(bodies[1]["withheld_error_mm"])
                                        )
                                    )
                                )
                            }
                            if child_points < 8
                            else {}
                        ),
                        "Attachment separation (mm)": float(
                            np.linalg.norm(joints[0] - joints[1])
                        ),
                        "Joint position error (mm)": float(
                            np.linalg.norm(np.mean(joints, axis=0) - known_joint)
                        ),
                    },
                )
            )
    for name, method in methods.items():
        withheld = [
            error
            for frame in method["frames"]
            for error in frame["bodies"][1]["withheld_error_mm"]
        ]
        method["summary"] = {
            "Maximum attachment separation (mm)": max(
                f["diagnostics"]["Attachment separation (mm)"] for f in method["frames"]
            ),
            "Landmark error vs known RMS (mm)": float(
                np.sqrt(
                    np.mean(
                        [
                            b["truth_rms"] ** 2
                            for f in method["frames"]
                            for b in f["bodies"]
                        ]
                    )
                )
            ),
        }
        if withheld:
            method["summary"]["Withheld child landmarks RMS (mm)"] = float(
                np.sqrt(np.mean(np.square(withheld)))
            )
        method["settings"] = dict(
            child_observed_ids=list(range(child_points)),
            child_withheld_ids=list(range(child_points, 8)),
            temporal="none",
            loss="squared",
            attachment=(
                "shared world point (exact)" if name == "connected" else "unconstrained"
            ),
            initialization=(
                "independent fits, mean attachment point"
                if name == "connected"
                else "identity at origin"
            ),
        )
        method["objective"] = (
            "Selected frame: 0.5 × summed squared landmark residuals (mm²). Attachment coincidence is exact, with no penalty weight."
            if name == "connected"
            else "Two separate landmark objectives. No combined iteration history."
        )
    return dict(
        parameters=dict(noise=noise, seed=seed, child_points=child_points),
        times=times.tolist(),
        methods=methods,
    )
