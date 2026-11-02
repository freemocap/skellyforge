# Synthetic object image tooling

## Calibrated temporal replay fixture

`temporal_renderer.py --mesh <source.stl> --features <features.json> --output <new-directory>`
uses the same CPU renderer to generate 240 timestamps at 30 Hz and four fixed
world cameras. One object follows a smooth six-degree-of-freedom trajectory.
Camera transforms are derived as `inverse(object_to_world) @ camera_to_world`;
the fixed world light is also transformed into the object's frame. Material,
camera backgrounds and sampling seeds stay constant through time.

The output includes four FFV1 MKV videos, each independently decoded and checked
for exact RGB identity against its 240 PNG frames. `ground_truth.npz` stores
frame/camera/keypoint projections and visibility, fixed camera intrinsics and
poses, object poses, exact `frame_index / 30` timestamps, and feature identities.
The manifest records hashes and the same image-edge pixel convention described
below. This independent temporal fixture is not the reserved final dataset.
No camera capture, Tracker import, training or GPU rendering is involved.

The explicit camera-transform override takes a separate branch from preview
camera construction; it does not calculate and discard a preview pose. An
existing `ffmpeg` executable is required for lossless video output; no package
installation is performed. Motion blur, sensor noise, rolling shutter and
physical interaction with composited backgrounds are not simulated.

## Authorized production generation

The current production target supersedes the preview-only 2560-image proposal:
20 roughly equal-area upper-hemisphere canonical reference views, 200 varied
training examples per anchor (4000 training images), 400 independently sampled
development images and 400 reserved final images. Canonical references are
excluded from all learning/evaluation groups. Ten percent of the training
examples use supplementary full-sphere directions; the others continuously
perturb their anchor directions. Roll, framing, distance, material, color,
illumination and background continue to vary independently.

`select_geometry_features.py` preserves the original 15 features and selects
35 additional fixed mesh points, with a nested 100-point definition for the
next comparison. It computes adjacent-face normal angles, retains crease or
boundary vertices at a 20-degree threshold, rejects candidates visible from
fewer than two of 40 probe cameras, and applies visibility-weighted farthest
point sampling. No image-dependent point selection or uniform point grid is
used. Repetitive/symmetric features can still be ambiguous; mirror-partner
indices and that limitation are saved explicitly. DLC does not enforce rigid
geometry merely because its labels came from a rigid object.

Production requires at least **100 distinct photographic source images**, with
at least 80 training, 10 development and 10 final sources. Crops or color variants
do not count as distinct sources. Source identities, related-scene families and
identical image bytes cannot cross splits. The acquired manifest carries actual
license metadata, attribution, download/source URLs, hashes, dimensions and
duplicate-review evidence. MediaWiki file metadata assists verification; the
software/API license is not the photograph license. [@mediawikiImageInfo]

The earlier three-photo production attempt was stopped and preserved before
training. It is explicitly superseded and must not be used as the production
dataset. The corrected runner rejects undersized photo pools before rendering,
snapshots source/configuration/provenance, uses a bounded four-image decoded
photo cache, and saves progress every 25 images. Existing output directories
are never overwritten.

After generation, `validate_production.py` checks all images/array hashes,
dimensions, feature ordering, recomputed projections, split isolation and photo
coverage. It creates a training-only gallery. `finalize_production.py` then
exposes one root `manifest.json`, `labels.npz` and `split_membership.json` with
4800 dense image rows. Image paths reference the existing split directories;
images are not copied. Membership keys are `training`, `development` and
`final_evaluation`. The canonical 20 remain separate. Final images receive
automated integrity checks only; no final contact sheet is produced or viewed.

This generation phase does not launch GPU training. The generic DLC tracker
remains separate from this STL/geometry dataset tooling.

## Current randomized preview

The original fixed-color dataset below remains preserved diagnostic evidence.
The user rejected its limited diversity for further training. The current
`randomized_renderer.py` adds full-sphere viewing directions, full camera roll,
distance/framing, color, diffuse/rough-plastic materials, colored illumination,
procedural backgrounds and reviewed public photographs. It still consumes a
generic mesh and explicit feature definition; Benchy is the initial fixture,
not a special-purpose DLC runtime tracker.

```powershell
python -B tools/synthetic_training/randomized_renderer.py --mesh .test-artifacts/benchy/source/3DBenchy.stl --features tools/synthetic_training/benchy_features.json --output .test-artifacts/benchy/randomized-preview-v5 --image-count 36 --samples-per-pixel 16 --photo-directory .test-artifacts/benchy/background-photos --photo-manifest tools/synthetic_training/photo_sources.json
```

The downloaded photographs are optional local assets, not bundled runtime data.
Their original bytes must match `photo_sources.json`. The three preview sources
have explicit CC0 image licenses, named authors, original URLs, dimensions,
hashes, attribution and review dates. This is an image-specific license check,
not an assumption based on Wikimedia's general site license.
[@commonsInteriorBackground] [@commonsForestBackground] [@commonsLandscapeBackground]

Every source photograph is assigned to one split. Different crops of the same
photograph cannot cross splits. The crop rectangle is saved per image, and the
photograph is decoded from sRGB into linear RGB before compositing. Procedural
backgrounds include independent noise, multiscale gradient-lattice noise with
quintic interpolation, and generated floor/panel graphics. These graphics are
not claimed to be photorealistic scenes or AI-generated images.

The initial AOV color/depth renderer was measured against ordinary RGBA rendering
on the same view at four samples/pixel: 5.832 s versus 0.0232 s. The current path
therefore uses ordinary RGBA with hidden direct emitters and native Embree array
queries for dense pixel-centre depth. Antialiased coverage comes from the renderer;
RGB is linear and premultiplied. The composite is
`foreground_rgb + (1 - coverage) * background_rgb`. Holes and partial silhouette
coverage preserve the background. The saved depth is physical ray distance in mm
at the pixel centre, with zero meaning no hit. Fractional coverage can legitimately
coexist with a zero centre-ray depth. No external occlusion or background contact
shadow is synthesized. [@mitsubaForegroundAov] [@embreexArrayQueries]

The inspected v5 preview contains 36 images at 640×480, including nine photo
composites. It took 19.92 seconds through hashes/manifest preparation, of which
4.93 seconds were render calls. All image and depth hashes were verified.
501 in-frame projections matched renderer ray directions within 1.69e-7 vector
norm. The 540 features include 202 ray-visible and 338 ray-occluded targets;
39 are outside the image. Visibility and projection validity remain separate.
Seventeen focused tests pass, including existing geometric projection/raster
checks and new depth, compositing, procedural reproducibility, photo color/crop,
and group-isolation checks.

The three preview labels (`training`, `development`, `final_evaluation`) exercise
split plumbing only. **All preview images have been inspected; none constitutes
an untouched final benchmark.** A full experiment must reserve fresh pose and
background seeds and unseen final source photographs. Original datasets and
models are not overwritten.

### Larger dataset proposal — not executed

Use 640 independent pose groups with four appearance variants each: 512 groups
for training (2048 images), 64 for development (256 images), and 64 for final
evaluation (256 images). Each split spans the same full orientation/framing
domain; appearance variants of a pose stay together. Use independent source-photo
pools for each split, not merely independent crops. Before generation, acquire
and verify a substantially broader photo pool (for example 20/5/5 source photos)
and reserve final source identities. The current three photographs establish
working photo compositing, not adequate real-world scene diversity.

Sample viewing directions uniformly in azimuth and sin(elevation), with uniform
roll. Stratify scale, image position and material/background categories within
each split. Record nearest cross-split pose distances and reject exact/near-copy
pose groups; do not claim random image shuffling prevents leakage. The current
preview balances twelve directional strata and does not yet implement that larger
group sampler. Preserve the original 30-view baseline as a separate diagnostic.

The observed preview rate extrapolates to roughly 24 minutes for 2560 images;
allow 20–40 minutes pending an actual larger benchmark. This is an estimate, not
measured full-run performance or training time. Keypoint-count experiments
(15/50/100) and training are separate steps. No larger generation or training is
launched by the preview command.

Remaining limits include only three preview photos, no external occluders,
distortion, motion blur, texture/material library or physically coupled shadows
on photographic backgrounds. No synthetic-to-real tracking-quality claim follows
from this renderer preview.

## Original geometry baseline

This optional, standalone Forge tool renders a fixed mesh and projects its named
object features. It does not import Tracker, Core or the Forge runtime. Training
and DLC-specific conversion belong to Tracker. The initial dataset is an
integration baseline, not evidence that a trained model generalizes to webcams.

Use a separate Python 3.12 environment with the exact optional packages in
`requirements.txt`. The verified Windows CPU path uses Mitsuba `scalar_rgb` and
Embree array queries. No GPU, LLVM installation or runtime dependency change is
required. A Mitsuba LLVM initialization warning does not prevent this explicit
scalar CPU variant from rendering. See [renderer evidence](BENCHY_REVIEW.md).

```powershell
python -B tools/synthetic_training/render_dataset.py --mesh PATH/3DBenchy.stl --features tools/synthetic_training/benchy_features.json --output .test-artifacts/benchy/dataset-v1 --image-count 160 --samples-per-pixel 16
python -B tools/synthetic_training/review_dataset.py .test-artifacts/benchy/dataset-v1
python -B -m unittest discover -s tools/synthetic_training -p test_render_dataset.py -v
```

The output directory must not already exist: this tool does not overwrite an
existing dataset. Verify your output path is ignored before generating artifacts.
The source STL remains unchanged; its hash must match the approved definition.

## Point selection provenance

The original single-part Benchy came from the official download site's link to
CreativeTools/3DBenchy; its exact revision/hash and nominal dimensions are in
[the review](BENCHY_REVIEW.md). The 15 identities were approved as an initial
baseline. They were **not** downloaded labels, and they were **not** discovered
by a general automatic feature-selection algorithm.

| Point group | Authored decision | Computed position |
| --- | --- | --- |
| Four roof corners | Select the upper roof plane band, approximately `z = 38.977 + 0.09628*x` in source mm, tolerance 0.02 mm, with `36 < z < 42` | Maximize four fixed XY diagonal support directions over vertices in that band |
| Four front-window corners | Identify the aperture and author approximate upper/lower, port/starboard corner coordinates | Select the nearest original mesh vertex to each target |
| Upper bow tip | Author the approximate fixed upper bow tip coordinate | Select its nearest original mesh vertex |
| Two stern rim shoulders | Author approximate upper transom corner coordinates | Select their nearest original mesh vertices |
| Four chimney rim points | Author four cardinal positions on the top outer rim | Select their nearest original mesh vertices |

Thus geometric calculations provide reproducible surface coordinates, while the
semantic regions and seed positions were selected manually. The checked-in
`benchy_features.json` freezes exact coordinates and ordering. No per-image
silhouette selection is performed. Circular chimney rim identities can remain
visually ambiguous despite having exact coordinates.

The original 15-point baseline did not perform automatic feature discovery.
The production `select_geometry_features.py` tool now extends those approved
identities using mesh crease/boundary candidates, spatial separation and view
coverage, as described above. Its additional points are geometric identities,
not automatically recognized semantic boat parts.

## Export contract

`images/` contains RGB PNGs. `labels.npz` has only dense numeric/string arrays;
it loads with `allow_pickle=False`. The first axis is image order and the feature
axis always follows `feature_names`. No absent/occluded point is silently dropped.

| Array | Layout and meaning |
| --- | --- |
| `image_names` | `[N]` relative image paths |
| `feature_names` | `[K]` fixed ordered identities |
| `projected_xy_pixels` | `[N,K,2]` geometric projections, including occluded/out-of-frame points |
| `visibility` | `[N,K]` whether the camera-to-feature segment is unoccluded |
| `projection_valid` | `[N,K]` finite, inside near/far depth and image rectangle |
| `feature_positions_mm` | `[K,3]` fixed object-frame coordinates |
| `camera_to_object` | `[N,4,4]` right-handed camera-to-object transform, column-vector convention |
| `camera_intrinsics` | `[N,3,3]` intrinsics matching saved projections |
| `image_seeds`, `ambient_values`, `light_intensities` | `[N]` exact per-image appearance settings |
| `view_targets_mm` | `[N,3]` camera look targets in object coordinates |
| `azimuth_radians`, `elevation_radians`, `camera_distances_mm` | `[N]` sampled camera pose parameters |

Camera axes are +X image-right, +Y image-down, +Z forward. Object axes are +X
starboard, +Y bow, +Z up. The object pose stays identity; cameras move around it.
Source-to-object rotation and origin are embedded in the manifest. Millimetres
and the separate 60 mm authored reference unit are never silently interchanged.

**Pixel coordinates use the top-left image edge as zero; pixel centres are at
half-integers.** The rendered-centroid test samples `(column+0.5,row+0.5)` to check
this convention. A consumer using integer pixel-centre coordinates must explicitly
subtract 0.5 from both coordinate components and principal-point coordinates at
its boundary; this is not a camera rotation or image flip.

Visibility is geometric, not model confidence or a training loss mask. The finite
ray ends `endpoint_tolerance_mm` before the feature to avoid surface self-hits;
the explicit default is 0.03 mm for this mesh. Out-of-frame and occlusion states
remain independent. The Tracker adapter owns the DLC label/mask convention.

`manifest.json` records original/feature/image/label hashes, dependency pins,
image dimensions, settings and limitations. Per-image render time excludes scene
creation and PNG encoding. `generation_seconds_before_manifest_write` includes
imports, mesh/BVH construction, scene setup, rendering, encoding, labels and
hashing; it excludes only writing the final manifest and process startup.

## Original baseline split and limits

`split_membership.json` stores exact image indices and paths: training elevations
at most 45 degrees, evaluation elevations at least 55 degrees, intervening views
excluded. This is a deliberately harder held-out elevation test, not a random
same-distribution accuracy estimate. Pose duplicates/appearance variants at the
same elevation stay on the same side; a positive angular gap prevents nearby
cross-split views. The baseline currently generates one appearance per pose.

The initial render distribution varies azimuth, elevation, distance, look target,
ambient brightness and point-light intensity. It does not yet vary material,
background texture, camera distortion, motion blur, external occluders or object
roll. Those limits matter for synthetic-to-real transfer. A successful synthetic
training/evaluation run alone cannot establish real-camera tracking quality.
