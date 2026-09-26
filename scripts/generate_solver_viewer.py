"""Generate the offline solver lab: cube experiments and a two-segment linkage."""

import itertools
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts.viewer_assets import vendored_scripts
from scripts.solver_sequence_experiment import sequence_experiment
from scripts.solver_lab_data import experiments


def experiment(*, noise, missing, outlier, seed):
    local = np.array(list(itertools.product([-100.0, 100.0], repeat=3)))
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    q = np.r_[np.cos(np.pi / 8), axis * np.sin(np.pi / 8)]
    truth = Rotation.from_quat(q, scalar_first=True).apply(local) + [60, 30, 90]
    observed = truth + np.random.default_rng(seed).normal(0, noise, truth.shape)
    observed[0] += [outlier, 0, 0]
    count = 8 - missing
    result = _native.fit_rigid(
        local=local[:count].tolist(),
        observed=observed[:count].tolist(),
        quaternion=[1.0, 0.0, 0.0, 0.0],
        translation=[0.0, 0.0, 0.0],
    )
    fitted = (
        Rotation.from_quat(result.quaternion, scalar_first=True).apply(local)
        + result.translation
    )
    return dict(
        units="mm",
        quaternion_order="wxyz",
        ceres_version=_native.ceres_version,
        native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        settings=dict(
            loss="squared",
            linear_solver="DENSE_QR",
            function_tolerance=1e-12,
            gradient_tolerance=1e-12,
            parameter_tolerance=1e-12,
        ),
        initial_quaternion=[1.0, 0.0, 0.0, 0.0],
        initial_translation=[0.0, 0.0, 0.0],
        noise=noise,
        missing=missing,
        outlier=outlier,
        seed=seed,
        local=local.tolist(),
        truth=truth.tolist(),
        observed=observed[:count].tolist(),
        fitted=fitted.tolist(),
        quaternion=result.quaternion,
        translation=result.translation,
        reference_quaternion=q.tolist(),
        reference_translation=[60, 30, 90],
        costs=result.costs,
        converged=result.converged,
        report=result.report,
        seconds=result.seconds,
        residuals=np.linalg.norm(fitted[:count] - observed[:count], axis=1).tolist(),
        truth_rms=float(np.sqrt(np.mean(np.sum((fitted - truth) ** 2, axis=1)))),
        rotation_error_degrees=float(
            np.degrees(2 * np.arccos(np.clip(abs(np.dot(q, result.quaternion)), 0, 1)))
        ),
    )


def main():
    cases = [
        experiment(noise=n, missing=m, outlier=o, seed=s)
        for n, m, o, s in itertools.product([0, 1, 5, 15], [0, 3], [0, 80], [7, 42])
    ]
    sequences = [
        sequence_experiment(noise=n, missing=m, outlier=o, seed=s, moving=moving)
        for n, m, o, s, moving in itertools.product(
            [0, 1, 5, 15], [0, 3], [0, 80], [7, 42], [False, True]
        )
    ]
    render(cases=cases, sequences=sequences)


def render(*, cases, sequences):
    render_experiments(bank=experiments(cases, sequences))


def render_experiments(*, bank):
    folder = Path(__file__).parent
    template = (folder / "solver_viewer.html.template").read_text(encoding="utf-8")
    template = template.replace(
        "__STYLE__", (folder / "solver_viewer.css").read_text(encoding="utf-8")
    )
    assets = (
        vendored_scripts()
        + "\n"
        + (folder / "vendor" / "plotly-basic-2.35.2.min.js").read_text(encoding="utf-8")
    )
    page = template.replace("__ASSETS__", assets).replace(
        "__EXPERIMENTS__", json.dumps(bank, allow_nan=False)
    )
    page = page.replace(
        "__VIEWER__",
        "\n".join(
            (folder / name).read_text(encoding="utf-8")
            for name in [
                "solver_scene.js",
                "solver_layout.js",
                "solver_time_series.js",
                "solver_map.js",
                "solver_problem.js",
                "solver_viewer.js",
            ]
        ),
    )
    path = folder / "solver_viewer.html"
    path.write_text(page, encoding="utf-8")
    print(f"{path.resolve()} - {len(bank)} experiments")


if __name__ == "__main__":
    main()
