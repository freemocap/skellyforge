# Axial position / robust keypoint fit: validation checkpoint

Validated 2026-09-27. The owner approved the sample recording visually. This is
a SkellyForge experiment checkpoint, not a production pipeline integration.

## Configuration

Three active frames with two fixed history frames. Independent sacrolumbar,
thoracic and cervical lengths, with existing 20:20:12 soft proportions (50 mm
scale) and 50% reference-length residual scale. No fixed cervical length.
SC forward offset 75%, lowered by 10% of thoracic reference length; relaxed
shoulder linkages; existing chest-line, axis and twist preferences retained.

Explicit Ceres XYZ residuals keep pelvis_origin, chest_center, neck_center and
craniocervical_junction near their saved post-hoc landmark positions (5 mm scale).
These are correlated geometric preferences, not independent measurements.
Direct keypoint residuals use Huber loss with a 30 mm physical transition.
Its scaling includes the existing time weight, preserving the threshold in mm.

An independent acceptance check rejects any checked axial landmark more than
50 mm from its saved position. This is an experimental acceptance limit, not
a hard Ceres spatial bound or anatomical tolerance. The viewer distinguishes
failed geometry from numerical convergence. No output poses are clipped or moved
after solving, and no frames are excluded.

## Both complete recordings

| Metric | Test (6 FPS) | Sample (30 FPS) |
| --- | ---: | ---: |
| Frames | 222 | 1108 |
| Converged windows | 220 / 220 | 1106 / 1106 |
| Ceres time | 81.02 s | 65.79 s |
| Window controller wall time | 83.04 s | 80.40 s |
| Median iterations / window | 27 | 4 |
| Maximum pelvis position error | 20.30 mm | 35.42 mm |
| Maximum chest position error | 1.19 mm | 11.91 mm |
| Maximum neck position error | 8.47 mm | 39.36 mm |
| Maximum head-attachment position error | 15.99 mm | 30.76 mm |

Both pass the all-frame position acceptance check. Source recording hashes and
native binary identity are checked; configuration comparison excludes the
person-scale values naturally estimated separately for each recording.
Timers exclude viewer preparation unless explicitly labeled otherwise.
The decimated sequence needs substantially more iterations per window. Larger
inter-frame motion is a plausible explanation, not a separately proven cause.

## Length variation and temporal limits

| Segment | Test length mean ± SD | Sample length mean ± SD | Length RMS difference at shared times |
| --- | ---: | ---: | ---: |
| Sacrolumbar | 272.0 ± 18.5 mm | 272.8 ± 20.1 mm | 4.39 mm |
| Thoracic | 271.4 ± 18.0 mm | 271.9 ± 18.2 mm | 3.13 mm |
| Cervical | 211.6 ± 25.9 mm | 208.4 ± 21.7 mm | 22.57 mm |

Length variation largely resembles variation in saved landmark endpoint spans:
their SDs are 17.9/17.9/25.2 mm for test and 18.0/18.0/22.3 mm for sample.
That does not establish anatomical accuracy or prove the absence of jitter.
At sample frames 1075–1076, cervical length changes by 76.32 mm over 33 ms;
the saved endpoint span changes by 80.46 mm. At test frames 193–194, cervical
length changes by 104.90 mm over 167 ms; the saved span changes by 99.76 mm.
Sample thoracic length also changes by 41.45 mm at frames 965–966. These are
remaining temporal/trajectory-quality concerns, not hidden by overall RMS.

Three active frames cover different durations at different frame rates. These
results do not establish frame-rate invariance. Independent calibration, filtering
(sample enabled, test disabled) and scale estimation also differ. Common-time
comparison interpolates scalar lengths; it does not compare different world bases
or use either reconstruction as ground truth.

The bad sample tail ankle target (Z approximately -917 mm) remains in the saved
data. Robust loss reduces its influence rather than rewriting it. The geometric
position priors can still follow errors in their own derived landmark targets.

## Reproduce and review

From the SkellyForge repository:

```powershell
poe solver-viewer-spine-positions --recording test
poe solver-viewer-spine-positions --recording sample
poe solver-validate-spine-positions
```

Sample generation currently uses `build/fixed_neck/fixed_neck.json` as its
comparison baseline; test generation does not require that older fit. Each reads
the prepared recording under `~/freemocap_data/testing/prepared/`; neither invokes
FreeMoCap, changes its environment, or regenerates upstream processing.

- Test viewer: http://127.0.0.1:8773/recording_test_spine_positions.html
- Sample viewer: http://127.0.0.1:8773/recording_spine_positions.html
- Full generated metrics: `build/spine_positions/validation.json`.

40 focused Python tests pass, including physical-time metric checks at 6/30/120
FPS and quaternion sign invariance. The test viewer check exercises all 222
frames. Its outdated assumption that frame 221 has no keypoint targets was
replaced with an explicit warning-rendering fixture; the rebuilt recording has
complete processed targets. The previous native CTest checks also passed; this
validation stage makes no native implementation changes.

## Handoff

Ready to commit as the validated experimental fit checkpoint on
`development-streaming`. Keep the accepted configuration unchanged. The next
work item is deliberate temporal regularization and upstream target-quality
review, with the current fit retained as the comparison. Do not call the solver
production-ready, frame-rate invariant, or temporally finished on this evidence.
No FreeMoCap dependency refresh is needed to continue development in Forge.
