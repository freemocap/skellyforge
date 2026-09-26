"""Compare a fixed linkage with a bounded scalar displacement, with optional missing middle observations."""

import numpy as np
from scripts.solver_chain_experiment import chain_experiment


def displacement_experiment(*, noise, extension, gap=0):
    methods = {}
    for name, enabled in [("fixed", False), ("displacement", True)]:
        run = chain_experiment(
            noise=noise, gap=gap, extension=extension, allow_displacement=enabled
        )
        method = run["methods"]["temporal"]
        frames = method["frames"]
        for i in run["gap_frames"]:
            frames[i]["observability"] = (
                "Middle landmark residual blocks are absent. In this axial geometry, "
                "root and distal attachment positions determine the middle Z axis and "
                "160 mm + displacement distance, but not middle roll about Z. "
                "Temporal residuals select a roll preference; convergence does not prove roll recovery."
            )
        method["summary"].update(
            {
                "All landmark RMS vs known (mm)": float(
                    np.sqrt(
                        np.mean(
                            [b["truth_rms"] ** 2 for f in frames for b in f["bodies"]]
                        )
                    )
                ),
                "Displacement RMS error vs known (mm)": float(
                    np.sqrt(
                        np.mean(
                            [
                                (f["displacement"] - f["reference_displacement"]) ** 2
                                for f in frames
                            ]
                        )
                    )
                ),
                "Frames at displacement bound": sum(
                    abs(f["displacement"]) >= 40.0 - 1e-6 for f in frames
                ),
            }
        )
        method["objective"] = (
            "One chain Problem with scalar displacement blocks at the second linkage, bounds +/-40 mm, zero-displacement prior residuals and displacement acceleration residuals. All segments retain fixed local geometry."
            if enabled
            else "Fixed-linkage baseline: no displacement parameter blocks. Same observations and other residual scales as the displacement model."
        )
        methods[name] = method
    return dict(
        parameters=dict(noise=noise, extension=extension, gap=gap),
        gap_frames=run["gap_frames"],
        gap_times=run["gap_times"],
        times=run["times"],
        methods=methods,
        displacement_series=True,
    )
