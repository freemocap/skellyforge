"""Three world quaternion blocks plus one root position per timestamp."""

import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts.solver_linkage_experiment import rotation


def chain_inputs(*, noise, gap, extension=0.0, branching=False):
    local = np.array(
        list(itertools.product([-25.0, 25.0], [-25.0, 25.0], [-80.0, 80.0]))
    )
    parent = np.array([[0.0, 0.0, 80.0], [0.0, 0.0, 80.0]])
    child = -parent
    if branching:
        parent = np.array([[0., -45., 80.], [0., 45., 80.]])
    times = np.linspace(0, 2, 41)
    missing = set(range(20 - gap // 2, 21 + gap // 2)) if gap else set()
    rng = np.random.default_rng(7)
    records = []
    for i, time in enumerate(times):
        phase = 2 * np.pi * time / 2
        # A local bend centered in the gap prevents initialization from being an
        # exact answer: distal observations must influence the middle quaternion.
        bend = 0.45 * np.exp(-(((time - 1) / 0.16) ** 2))
        rotations = [
            rotation([1, 2, 0], 0.2 * np.sin(phase)),
            rotation([1, 0.2, 0], 0.5 + 0.3 * np.sin(phase) + bend),
            rotation([0, 1, 0.2], -0.4 + 0.25 * np.cos(phase)),
        ]
        if branching:
            rotations[1] = rotation([1, 0, 0], 0.8 + 0.3 * np.sin(phase) + bend)
            rotations[2] = rotation([1, 0, 0], -0.8 + 0.25 * np.cos(phase))
        displacement = extension * np.sin(np.pi * time / 2) ** 2
        translations = [np.array([30 * np.sin(phase), 15 * np.cos(phase), 130.0])]
        for b in range(1, 3):
            parent_index = 0 if branching else b - 1
            translations.append(
                translations[parent_index]
                + rotations[parent_index].apply(
                    parent[b - 1] + ([0, 0, displacement] if b == 2 else np.zeros(3))
                )
                - rotations[b].apply(child[b - 1])
            )
        truth = np.array([r.apply(local) + t for r, t in zip(rotations, translations)])
        observed = truth + rng.normal(0, noise, truth.shape)
        records.append(
            dict(
                displacement=float(displacement),
                rotations=rotations,
                translations=translations,
                truth=truth,
                observed=observed,
            )
        )
    return local, parent, child, times, missing, records


def chain_experiment(*, noise, gap, extension=0.0, allow_displacement=False, branching=False):
    local, parent, child, times, missing, records = chain_inputs(
        noise=noise, gap=gap, extension=extension, branching=branching
    )
    observations = [
        [
            record["observed"][b].tolist() if b != 1 or i not in missing else []
            for b in range(3)
        ]
        for i, record in enumerate(records)
    ]
    result = _native.fit_chain_sequence(
        local=[local.tolist()] * 3,
        observed=observations,
        parent_attachments=parent.tolist(),
        child_attachments=child.tolist(),
        times=times.tolist(),
        position_scale=10.0,
        linear_acceleration_scale=3000.0,
        angular_acceleration_scale=20.0,
        allow_displacement=allow_displacement,
        parent_indices=[0, 0] if branching else [0, 1],
    )
    frames = []
    for i, record in enumerate(records):
        bodies = []
        for b in range(3):
            q, t = result.quaternions[i][b], result.translations[i][b]
            iq, it = result.initial_quaternions[i][b], result.initial_translations[i][b]
            fitted = Rotation.from_quat(q, scalar_first=True).apply(local) + t
            initial = Rotation.from_quat(iq, scalar_first=True).apply(local) + it
            bodies.append(
                dict(
                    local=local.tolist(),
                    truth=record["truth"][b].tolist(),
                    observed=observations[i][b] or [None] * 8,
                    fitted=fitted.tolist(),
                    initial=initial.tolist(),
                    quaternion=q,
                    translation=t,
                    initial_quaternion=iq,
                    initial_translation=it,
                    reference_quaternion=record["rotations"][b]
                    .as_quat(scalar_first=True)
                    .tolist(),
                    reference_translation=record["translations"][b].tolist(),
                    residuals=(
                        np.linalg.norm(fitted - record["observed"][b], axis=1).tolist()
                        if observations[i][b]
                        else [None] * 8
                    ),
                    truth_rms=float(
                        np.sqrt(
                            np.mean(np.sum((fitted - record["truth"][b]) ** 2, axis=1))
                        )
                    ),
                    initial_truth_rms=float(
                        np.sqrt(
                            np.mean(np.sum((initial - record["truth"][b]) ** 2, axis=1))
                        )
                    ),
                )
            )
        separations = []
        for b in range(2):
            parent_index = 0 if branching else b
            a = (
                Rotation.from_quat(result.quaternions[i][parent_index], scalar_first=True).apply(
                    parent[b]
                )
                + result.translations[i][parent_index]
            )
            c = (
                Rotation.from_quat(
                    result.quaternions[i][b + 1], scalar_first=True
                ).apply(child[b])
                + result.translations[i][b + 1]
            )
            separations.append(float(np.linalg.norm(a - c)))
        frames.append(
            dict(
                time=float(times[i]),
                bodies=bodies,
                root=result.roots[i],
                displacement=result.displacements[i],
                reference_displacement=record["displacement"],
                costs=result.costs,
                converged=result.converged,
                seconds=result.seconds,
                report=result.report,
                diagnostics={
                    "First attachment separation (mm)": separations[0],
                    "Second attachment separation (mm)": separations[1],
                    "Middle landmark residual blocks": 0 if i in missing else 8,
                    "Middle fitted error vs known (mm)": bodies[1]["truth_rms"],
                    "Middle initialization error vs known (mm)": bodies[1][
                        "initial_truth_rms"
                    ],
                },
                observability=(
                    "Middle landmarks are absent. Distal residuals constrain the middle attachment vector; temporal residuals provide a preference for its unobserved roll."
                    if i in missing
                    else ""
                ),
            )
        )
    if allow_displacement or extension:
        for i, frame in enumerate(frames):
            frame["diagnostics"]["Second linkage displacement span (mm)"] = frame[
                "diagnostics"
            ].pop("Second attachment separation (mm)")
            frame["diagnostics"]["Displacement parameter (mm)"] = result.displacements[
                i
            ]
            frame["diagnostics"]["Known synthetic displacement (mm)"] = records[i][
                "displacement"
            ]
            a = (
                Rotation.from_quat(result.quaternions[i][1], scalar_first=True).apply(
                    parent[1]
                )
                + result.translations[i][1]
            )
            b = (
                Rotation.from_quat(result.quaternions[i][2], scalar_first=True).apply(
                    child[1]
                )
                + result.translations[i][2]
            )
            expected = Rotation.from_quat(
                result.quaternions[i][1], scalar_first=True
            ).apply([0, 0, result.displacements[i]])
            frame["diagnostics"]["Second linkage equation error (mm)"] = float(
                np.linalg.norm(b - a - expected)
            )
    summary = {
        "Ceres parameter blocks": result.parameter_blocks,
        "Ceres residual blocks": result.residual_blocks,
    }
    if missing:
        for key, label in [
            ("truth_rms", "Fitted"),
            ("initial_truth_rms", "Initialization"),
        ]:
            summary[f"{label} middle gap RMS vs known (mm)"] = float(
                np.sqrt(np.mean([frames[i]["bodies"][1][key] ** 2 for i in missing]))
            )
    if branching:
        for i, frame in enumerate(frames):
            frame["observability"] = ("Branch A has no landmark residual blocks here. Its quaternion is selected by temporal residuals; sibling observations do not observe its rotation." if i in missing else "")
            for b, label in enumerate(["Parent", "Branch A", "Branch B"]):
                body=frame["bodies"][b]
                angle=(Rotation.from_quat(body["quaternion"], scalar_first=True).inv() * records[i]["rotations"][b]).magnitude()
                frame["diagnostics"][label + " angular error (degrees)"]=float(np.degrees(angle))
                frame["diagnostics"][label + " landmark error vs known (mm)"]=body["truth_rms"]
                body["angular_error_degrees"]=float(np.degrees(angle))
        for b, label in enumerate(["Parent", "Branch A", "Branch B"]):
            summary[label + " landmark RMS vs known (mm)"]=float(np.sqrt(np.mean([f["bodies"][b]["truth_rms"]**2 for f in frames])))
            summary[label + " angular RMS (degrees)"]=float(np.sqrt(np.mean([f["bodies"][b]["angular_error_degrees"]**2 for f in frames])))
        summary["Branch A gap angular RMS (degrees)"]=float(np.sqrt(np.mean([frames[i]["bodies"][1]["angular_error_degrees"]**2 for i in missing]))) if missing else 0.
        for frame in frames:
            for key in list(frame["diagnostics"]):
                if "Middle" in key: frame["diagnostics"][key.replace("Middle", "Branch A")]=frame["diagnostics"].pop(key)
        for key in list(summary):
            if "middle" in key: summary[key.replace("middle", "Branch A")]=summary.pop(key)
    method = dict(
        frames=frames,
        summary=summary,
        problem=dict(
            chain=True,
            parents=[0, 0] if branching else [0, 1],
            connected=True,
            temporal=True,
            acceleration=True,
            displacement=allow_displacement,
        ),
        settings=dict(
            position_scale_mm=10.0,
            linear_motion_scale=3000.0,
            angular_motion_scale=20.0,
            scale_units=["mm/s^2", "rad/s^2"],
            displacement_scale_mm=20.0,
            displacement_acceleration_scale=500.0,
            displacement_bound_mm=40.0,
            quaternion_basis="world wxyz",
            loss="squared",
            initialization="Independent segment fits. Missing middle quaternions initialized with Slerp between observed frames; all blocks then optimized.",
            costs_by_family=dict(
                landmarks=result.landmark_cost,
                root_acceleration=result.root_acceleration_cost,
                displacement_prior=result.displacement_prior_cost,
                displacement_acceleration=result.displacement_acceleration_cost,
                segment_angular_acceleration=result.angular_acceleration_costs,
            ),
        ),
        objective="One sequence Problem: ChainLandmarkResidual blocks depend on the root and all upstream quaternion blocks. Four acceleration residual blocks per interior timestamp. Both linkages are exact by forward calculation.",
    )
    return dict(
        parameters=dict(noise=noise, gap=gap),
        times=times.tolist(),
        methods={"temporal": method},
        gap_frames=sorted(missing),
        gap_times=(
            [float(times[min(missing)]), float(times[max(missing)])]
            if missing
            else None
        ),
    )
