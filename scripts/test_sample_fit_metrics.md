# Test versus sample fit metrics

Generate with `python -m scripts.compare_recording_fit_metrics`; view
http://127.0.0.1:8773/recording_test_sample_metrics.html.

Inputs are the accepted `sc_75_rest_50.json` test fit and sample
`tolerance_1e-06.json`. Both use 20:20:12, 75% SC forward, 50% rest scale,
three active frames and the same stopping tolerance. Neither fit is ground truth.

The comparison matches 222 timestamps (0–36.833333 seconds) without interpolation,
and includes 13,085 keypoint targets present in both fits. RMS is Euclidean point
error, duration-weighted with trapezoidal weights. Length statistics use the same
timestamps, including model predictions on frames without keypoint targets.
Interactive plots retain every original frame, so spikes between test-data
timestamps remain visible. Per-frame plot RMS uses each recording's available
targets; it is distinct from the matched-population table.

| Quantity (mm) | Test | Sample |
| --- | ---: | ---: |
| Shared keypoint-target RMS | 15.95 | 22.53 |
| Left shoulder target RMS | 14.34 | 17.13 |
| Right shoulder target RMS | 14.76 | 17.77 |
| Sacrolumbar length standard deviation | 19.1 | 46.8 |
| Thoracic length standard deviation | 19.1 | 51.2 |
| Cervical length standard deviation | 26.4 | 32.4 |
| Sacrolumbar RMS deviation from reference length | 26.6 | 55.1 |
| Thoracic RMS deviation from reference length | 27.9 | 61.5 |
| Cervical RMS deviation from reference length | 37.4 | 40.4 |
| Left shoulder linkage displacement RMS | 10.62 | 21.92 |
| Right shoulder linkage displacement RMS | 11.41 | 27.91 |
| SC midpoint height above shoulder midpoint: mean | 6.59 | 32.17 |
| SC midpoint height above shoulder midpoint: standard deviation | 27.92 | 40.83 |

SC height is measured along the hip-center-to-shoulder-center axis. No SC
keypoint is directly tracked, so this is geometry, not an SC measurement error.
The JSON and HTML table also include normalized reference-length RMS and
dimensionless rest-residual RMS, means, ranges, and each fit's reference lengths.

## Remaining investigation

The sample's saved calibrated-world descriptor records `body_reference`, anchored
to `skull`, with a nonidentity quaternion and translation. The posthoc pipeline
did invoke person alignment. Its body-reference fallback centers the selected
body region; it does not establish a floor. The viewer places its grid at Z=0,
which is inappropriate to present as a recovered floor for this result. Determine
why foot support was unavailable and obtain the desired alignment through the
pipeline, rather than adding a display-only rotation/translation.

Three active frames span 0.333 seconds at 6 Hz but 0.067 seconds at 30 Hz.
Free axial lengths explicitly disable the existing length-acceleration residual.
Their rest-length and ratio residuals remain enabled. Root, quaternion, and
shoulder-displacement acceleration residuals use timestamp differences; inspection
shows velocity differences weighted by 1/sqrt(midpoint_dt), consistent with a
time-integrated squared-acceleration penalty. Do not presume a missing FPS factor.

Separate calibration, filtering policy, reference lengths, and temporal support
confound this comparison. Next isolate sampling effects by using the same prepared
sample input at full rate and every fifth frame, preserving real timestamps.
Then compare equal-duration windows and/or explicit length temporal regularization
and stronger rest-length preferences. Those experiments and the alignment fix
are not implemented by this metrics change. Do not label the current sample fit
accepted. The discrepancies confirm that the test recording alone was insufficient.
