# Full sample recording performance experiment

2026-09-26. All runs use the selected 3-active-frame fit, 2 fixed history frames,
20:20:12 spine proportions, 75% SC forward, 50% rest-length residual scale,
posterior centerline preference, and relaxed shoulders. No residual weights,
geometry, frame sampling, or native build were changed between runs.

## Input and preparation

The installed sample has **1,108 frames at 30 Hz**, timestamps 0–36.9 seconds,
not 2,000 frames. The dataset definition and decoded video counts agree.
The test recording has 222 frames at 6 Hz, approximately the same duration.

The source sample folder contained videos and calibration but no prepared
Parquet. The existing FreeMoCap preparation command was run once:

    python -B -m freemocap.tests.prepare_recording_dataset --dataset sample --timeout 3600

It regenerated calibration, tracked the videos, triangulated, filtered, and
reconstructed in the dedicated testing space. No FreeMoCap source was edited.
The retained input is:

    ~/freemocap_data/testing/prepared/freemocap_sample_data/current/recordings/freemocap_sample_data/freemocap_sample_data_data.parquet

Preparation ran approximately 18:50:41–18:55 (local), separately from the solve
timings below. The full preparation log and ready manifest are retained beside
the prepared recording. Camera matching reported `poor` geometry fitness
(camera median reprojection errors approximately 1.12, 1.59, 1.74 pixels;
p90 approximately 4.54, 4.94, 5.44 pixels). This is a material input-quality
diagnostic, not proof of a specific cause of subsequent fit behavior.

Sample preparation enables filtering; test preparation disables it. Together
with separate calibration and scale estimates, this means comparing their
runtimes is not a controlled experiment in frame count alone.

## Single-run timings

All cases use a 200-iteration maximum and all 1,106 windows report convergence.
Tolerance is Ceres function tolerance, not a residual strength or spatial tolerance.

| Function tolerance | First window | Ceres seconds | Window processing seconds | Read + fit seconds | Target RMS mm | Iterations p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-6 | 1e-6 | 38.29 | 47.02 | 69.39 | 25.816 | 4 / 7 |
| 1e-5 | 1e-5 | 30.71 | 39.16 | 62.95 | 25.868 | 3 / 5 |
| 1e-4 | 1e-4 | 27.10 | 36.31 | 56.95 | 25.777 | 3 / 4 |
| 1e-5 | 1e-6 | 31.69 | 40.55 | 64.29 | 25.650 | 3 / 5 |

Ceres seconds sum optimization times. Window processing also includes window
construction, result transfer, and final full-recording objective evaluation.
Read + fit includes the recording adapter and frame-result construction; it
excludes the runner's initial inventory read, annotated-preview extraction,
context preparation, JSON writing, metrics, and HTML generation. These are not
end-to-end UI timings or repeated statistical benchmarks.

The selected 222-frame test-data case previously took 70.76 Ceres seconds.
The denser sample often needs only 3–5 iterations per window; the smaller motion
between adjacent frames is consistent with easier warm starts. This result does
not establish throughput for other recordings or a complete realtime pipeline.

## Fidelity comparison

Differences are against the sample's 1e-6 baseline, not ground truth. Position
metrics include all fitted model landmarks; rotation metrics include all segment
quaternions, using the shortest relative-quaternion angle (sign invariant).

| Case | Landmark difference median / p95 / max, mm | Quaternion difference median / p95, degrees |
| --- | --- | --- |
| 1e-5 | 3.30 / 57.90 / 1136.69 | 68.78 / 167.59 |
| 1e-4 | 3.81 / 44.67 / 1251.35 | 78.12 / 170.14 |
| 1e-5, strict first window | 2.94 / 34.28 / 1217.52 | 56.00 / 167.34 |

The largest positional differences occur near the recording's sparse/unobserved
tail. There are 27 frames without mapped keypoint targets. Such frames are
supported by model and temporal residuals; they are not measured validation.
The rotation differences are material even though many landmark positions and
the aggregate target RMS remain similar. None of the faster settings is an
established fidelity-preserving replacement.

The baseline itself needs visual review: cervical length reaches approximately
558 mm at frame 953, 563 mm at 1073, and zero at 1076. Maximum relaxed shoulder
separation is 130 mm. Its cervical-axis dot torso-axis diagnostic never becomes
negative; that diagnostic alone therefore does not certify a good fit.
Do not promote these sample results solely because every window converged.

## Adapter correction

Frame 1077 has two saved keypoints but no saved reconstructed landmark positions.
The model still has all landmark definitions. `frame_targets` incorrectly
required a saved reconstructed position to use an available directly mapped
keypoint. Targets now come directly from the saved keypoint channel; where a
saved landmark position exists it must still agree. No source coordinates are
modified or fabricated. Existing test-recording targets are unchanged. The
regression test verifies that removing saved reconstruction positions does not
remove valid keypoint targets, while disagreement still raises.

## Review and next step

Open http://127.0.0.1:8773/recording_sample_performance.html.
All four fits cover frames 0–1107 and use synchronized annotated previews.
The Fit column identifies tolerance and the strict-start variant. Use Solo,
overlay, and local axes; inspect frames around 950–980 and 1064 onward as well as
ordinary motion. Sample previews have their own cache and do not overwrite the
test viewer's previews.

Keep 1e-6 as the current stopping setting. The next performance work should
profile and reduce repeated Parquet decoding/result preparation and per-window
overhead without changing the objective or stopping rule. Separately review
sample input quality and the extreme fitted spine lengths before calling the
full-rate result accepted. No production defaults were changed.

Reproduce from SkellyForge:

    uv run poe solver-viewer-sample-performance
    python -m scripts.solver_sample_performance --resume
    python -m scripts.solver_sample_performance --publish-only

Four fixed-name caches and `metrics.json` live in `build/sample_performance/`.
`--resume` reuses completed cases; `--publish-only` does not solve again.
The prepared Parquet remains read-only during these experiments.

Validation: 15 focused Python tests passed; viewer checks passed for 4 × 1,108
sample frames and the existing 6 × 222 test comparison. Headless browser review
confirmed the sample page, annotated imagery, controls, and inventory render.
