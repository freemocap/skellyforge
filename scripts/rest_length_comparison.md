# Spine rest-length comparison

The existing length residual prevents the observed cervical reversal around
frame 194 in both new 75% SC-forward fits. This is evidence for this recording,
not a guarantee of robustness on other movements or people.

## What 75% SC forward means

`solver_shoulder_offsets.reference_positions` multiplies only the forward
component of the sternoclavicular attachments relative to neck center in the
thoracic local frame: 102.666 mm becomes 76.999 mm. Both profiles retain the
same 26.950 mm lowering, lateral spacing, clavicle lengths, and person scale.
This is not a 25% reduction of the entire Euclidean distance. The changed
attachment geometry enters Ceres and the viewer consistently; there is no
frame-dependent or display-only correction.

## Experiment

Full prepared 222-frame test recording; three active frames and two fixed
history frames; 20:20:12 length-ratio preference retained. No new bending
residual, length smoothing, or upper length bound. Lengths remain nonnegative.
The native `free_length_rest_prior` flag enables the existing LengthPriorResidual
independently of the free-length policy, which previously disabled it.

For each flexible segment the residual is:

    sqrt(frame_weight) * (length - reference_length) / (scale * reference_length)

Reference lengths are 272.080 mm sacrolumbar, 269.498 mm thoracic, and 206.570 mm
cervical. These saved reference lengths and the 20:20:12 proportion preference
are distinct, potentially competing preferences. A 25% scale gives four times
the squared-error penalty of a 50% scale for the same length deviation. Neither
percentage is a hard permitted range.

| SC forward | Rest scale | Ceres seconds | Read + fit seconds | Target RMS mm | Converged windows | Cervical axis dot torso axis at frame 194 |
| --- | --- | --- | --- | --- | --- | --- |
| 100% | Off | 80.24 | 86.59 | 15.964 | 219/220 | 0.938 |
| 100% | 50% | 84.39 | 92.46 | 15.968 | 220/220 | 0.941 |
| 100% | 25% | 71.60 | 77.83 | 15.950 | 220/220 | 0.937 |
| 75% | Off | 80.62 | 87.78 | 15.979 | 220/220 | -0.397 |
| 75% | 50% | 70.76 | 76.33 | 15.974 | 220/220 | 0.979 |
| 75% | 25% | 70.01 | 76.42 | 15.964 | 220/220 | 0.963 |

The diagnostic compares the fitted cervical local +Z axis with the normalized
hip-center-to-shoulder-center vector. Negative means the axes oppose each other;
this is a diagnostic for the observed fold, not an anatomical ground-truth test.
The original 75% case has three negative frames. Neither new 75% case has any;
their minimum dot products are 0.584 and 0.648, respectively. At frame 203 they
are 0.991 and 0.993, compared with 0.738 without the rest preference.
Target RMS alone barely distinguishes these configurations. The final six
frames have no mapped keypoint targets and must not be treated as measured
validation of the fit.

## Review and reproduce

Open http://127.0.0.1:8773/recording_rest_length_comparison.html while the existing
viewer server serves `scripts/`. Start at frames 190–205, then inspect the whole
recording. The Rest column distinguishes Off, 50%, and 25%. Details include the
settings and geometric diagnostic. Initially the original 75% and its 25%
rest-scale fit are overlaid. Use Solo to inspect each independently.

    uv run poe solver-viewer-rest-lengths
    python -m scripts.solver_rest_length_comparison --publish-only

The first command refits four cases and reuses the two saved controls. The second
only republishes cached results. `--resume` skips completed cases after an
interruption. Caches use fixed names under `build/rest_length_comparison/`.

Validation: 24 focused Python tests, three native CTest tests, and comparison
viewer checks for all six new-page fits and the existing processing, ratio,
and SC-offset pages passed. No production default has been selected by this
experiment. Review the weaker 50% preference first: it already prevents the
observed fold while applying less pressure toward the saved reference lengths.
