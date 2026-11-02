# Benchy geometry and renderer review

The user approved these 15 features as the initial baseline. This document records
the acquisition and renderer evidence; [the tooling README](README.md) explains
the authored seed choices, geometric calculations and current dataset contract.

The original single-part mesh was obtained through the official download page's
GitHub link, at commit `b542785f9642ab39db8a778a0ccc934b3ebd829a`.
Its SHA256 is `6ab57f1c3f8e86bc3cbd302c6fa6270acf06277c6335454e922419c25d42e97e`.
The source file is preserved unchanged in ignored `.test-artifacts/benchy/source/`.
[@benchyOfficialDownload] [@benchyOriginalMesh]

The mesh has 225706 triangles and bounding extents
`60.00100136 × 31.00399971 × 48.0` source units. Official nominal dimensions are
`60 × 31 × 48 mm`, supporting millimetres for this original-size mesh. The small
extent differences are consistent with mesh discretization/rounding; we do not
silently rescale to force the nominal numbers. STL itself does not encode units.
The current official license page says CC0 1.0 Universal. Keep attribution and the
retrieval record even though that page does not require attribution.
[@benchyOfficialDimensions] [@benchyOfficialLicense]

## Proposed feature identities and frame

Review `.test-artifacts/benchy/feature-review.png` alongside
`candidate_features.json` and `candidate_features.npz`. Fifteen proposed features
come from recognizable structures: four upper roof corners, four front-window
corners, the upper bow tip, two stern-rim shoulders and four top chimney-rim points.
Each is a fixed authored mesh point. The rounded roof corner positions are fixed
support points of the roof's upper surface, not silhouette extrema recomputed
for each view. Every chimney point is a surface rim point; no point at a hole
centre is assigned misleading surface visibility. Point count follows the selected
features and can change during review; it is not a system constant.

The source mesh frame is +X bow, +Y port, +Z up. Proposed Forge object axes are
+X starboard, +Y bow, +Z up. Origin is the centre of the bottom footprint's
bounding box at z=0; the source coordinates of that origin are recorded explicitly.
The proper rotation is `[[0,-1,0],[1,0,0],[0,0,1]]`, with determinant +1.
Authored geometry would use a 60 mm reference unit, while physical coordinates
remain explicitly millimetres. Reuse `build_rigid_marker_skeleton` for fitting;
no new object solver is needed here.

Preserve port/starboard identities rather than silently collapsing mirrored
views. Bow/stern and roof/chimney geometry distinguish fore/aft; side-ambiguous
or occluded images remain possible and must be measured as failures. Do not
apply horizontal-flip augmentation unless its feature permutation is explicit.
The four chimney cardinal points are fixed in the object frame, but a circular
rim alone does not make their identities visually distinguishable. They are
candidate validation targets, not a guarantee that image inference can recover them.
Occluded feature projections remain target labels. Validity and visibility are
separate arrays; neither is a synthetic model-confidence score.
Whether DLC's training adapters preserve the desired occlusion supervision must
be checked separately; geometric visibility does not establish training-mask behavior.

## Renderer recommendation and CPU proof

Use Mitsuba as an optional Forge tool dependency, isolated from Forge runtime and
Tracker's separate DLC environment. It accepts the mesh's dense buffers and an
explicit perspective sensor. This keeps geometry, projection and rendering in
Forge without creating Core-owned rendering logic or Blender-specific coupling.
[@mitsuba391Mesh] [@mitsubaPerspectiveSensor]

A task-local environment under `.test-artifacts/benchy/render-env` contains
Mitsuba 3.9.1, Dr.Jit 1.5.0, NumPy 2.4.6 and Pillow 12.3.0; the accompanying
`render-project/uv.lock` records the resolved environment. The feasibility script
selected `scalar_rgb` explicitly: three 640×480 views at 8 samples/pixel rendered
in approximately 0.05–0.07 seconds each on this machine (render call only).
Analytical projected directions agreed with sensor-sampled rays within 7.3e-8
vector norm. This is a smoke check, not a production performance benchmark or the
full subpixel label-registration test. LLVM initialization reported unavailable;
no GPU variant or driver modification was used.

The review-only script makes scalar visibility queries for fifteen points as a
reference. A separate feasibility check now verifies the production candidate:
one dense NumPy ray-array call to `embreex` 4.4.0.post1, whose compiled Cython loop
releases the GIL and traverses the native Embree BVH. This avoids Python
per-feature loops. The mesh arrays are uploaded once to both rendering and
visibility engines. [@embreexArrayQueries]

All 45 camera/feature visibility results match the scalar reference. The same
results hold after rotating into the proposed object frame and converting to the
60 mm reference unit. Five simple plane checks cover target clearance,
surface endpoints and near/far occlusion. Occlusion is tested on the finite
camera-to-target segment, ending 0.03 mm before the surface target in this proof.
Unbounded rays through exact corner/rim vertices can disagree on which surface
they hit *behind* the target; that is not evidence of an occluder in front of it.
Segment-limited hit distances agreed within 0.000031 mm. B2 must retain these
boundary tests and expose the scale-aware endpoint tolerance rather than bury it.

The second BVH construction plus initial query took 0.109 seconds and added an
observed 24.4 MB to process working set. A reused-BVH query of 45000 repeated
test rays took 0.0106 seconds. These figures include allocator effects and a small
repeated ray set; they do not establish full dataset throughput.

The alternative LLVM CPU backend was tested using an official LLVM 18.1.8 DLL
extracted into the experiment directory and a process-local library path. It
loaded but failed a tiny array computation with unresolved JIT symbols. No
system installer, driver or global PATH change was made. The recommended
Mitsuba-scalar plus Embree-array path does not require that LLVM runtime.
[@drjitCpuBackend] [@llvmPortable1818]

The subsequent generator passed sensor-ray and rendered-centroid checks, with
centroid agreement within 0.2 pixels. The first dataset contains 160 images:
95 training, 30 evaluation and 35 excluded guard-band views. Nine focused tests
pass. Variable external occluders, broader appearance variation and real-camera
transfer remain limitations; training belongs to the Tracker-owned next step.
