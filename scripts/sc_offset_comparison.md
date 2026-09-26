# SC forward-depth comparison

The owner selected three-frame fitting with sacrolumbar:thoracic:cervical ratios
20:20:12. That ratio is now the recording experiments' default.

This comparison uses the existing `lower_closer_sc` geometry profile: SC forward
offset becomes 75% of its previous value, approximately 102.7 to 77.0 mm for this
recording. Lateral positions and the 10%-of-reference-thoracic lowering remain
unchanged. The change applies to both SC landmarks and the central notch, with
matching clavicle attachments used by Ceres. It is not a display-only shift.
The complete neck-to-SC distance also includes lateral/axial components, so that
Euclidean distance is not reduced by exactly 25%.

All 222 frames are refitted from the existing prepared Parquet using three active
frames and two constant history frames, with function tolerance 1e-6. Spine
ratios, rigid clavicle lengths, keypoint targets, person scale and residual
strengths are held fixed. Videos and tracking/triangulation are not reprocessed.

Open <http://127.0.0.1:8773/recording_sc_offset_comparison.html>. Original and
reduced-depth fits are overlaid; use Solo to inspect either. The compact SC depth
column shows 100% versus 75%. Details retains the complete geometry settings.

From SkellyForge:

```powershell
uv run --no-sync poe solver-viewer-sc-offset
uv run --no-sync poe solver-viewer-sc-offset --publish-only
```

The new fit overwrites `build/sc_offset_comparison/forward_75.json`; the accepted
20:20:12 baseline is reused from `build/spine_ratio_comparison/neck_12.json`.
The saved reference recording and original fit are unchanged.

SC-midpoint height along the hip-to-shoulder axis changes from 12.8 to 3.2 mm
at frame 188, from 22.0 to -11.8 mm at frame 192, and from 31.7 to 79.7 mm at
frame 203. The requested smaller depth is achieved, but shoulder placement is
not uniformly improved. Inspect the raised-arm frames before adopting this
offset as a default; the original depth remains available for comparison.

The reduced-depth fit converged in all 220 windows: 80.6 seconds in Ceres, 87.8 seconds including reading and adapting the recording. Keypoint-target RMS is 15.979 mm.

Validation: 13 focused Python tests passed, including reference-geometry and
attachment consistency checks. Viewer geometry and playback are also checked.
The final six frames still lack mapped keypoint targets; their fitted poses are
not supported by measurements.
