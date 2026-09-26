"""Review an experimental Forge sequence fit against unchanged saved real data.

Run as a module from the Forge checkout. The producer's Parquet is read-only;
only scripts/recording_fit_viewer.html is overwritten. This is not a replacement
for production posthoc reconstruction or its saved outputs.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from scripts.recording_data import recording_path, read_recording, digest
from scripts.generate_real_skeleton_viewer import saved_segments
from scripts.viewer_assets import geometry_script, vendored_scripts
from scripts.recording_fit_geometry import rigid_target_checks, spine_bend_degrees
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    LandmarkTarget,
    RootPoseTolerances,
)
from skellyforge.core.skeleton.pose.fit_connected_sequence import (
    ConnectedFrameInput,
    MotionTolerances,
    fit_connected_sequence,
)

TARGET_NAMES = (
    "left_hip_socket",
    "right_hip_socket",
    "left_acromion",
    "right_acromion",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_ear",
    "right_ear",
    "nose",
)


def target_sources(model):
    """Verify direct saved correspondences; do not reimplement the tracker mapper."""
    sources = {}
    for mapping in model["mappings"]:
        for name, entry in mapping["entries"].items():
            if name not in TARGET_NAMES:
                continue
            if name in sources or not isinstance(entry, str):
                raise ValueError(f"Review requires one direct saved mapping for {name}")
            sources[name] = (mapping["prefix"] or "") + entry
    if set(sources) != set(TARGET_NAMES) or len(set(sources.values())) != len(sources):
        raise ValueError(
            "Review requires all selected targets with unique source keypoints"
        )
    if not set(sources.values()).issubset(model["tracker_keypoint_names"]):
        raise ValueError("Selected source is not a declared tracker keypoint")
    return sources


def observed_targets(record, sources, scale):
    targets = {}
    for name, source in sources.items():
        landmark = record["points"].get(name)
        keypoint = record["keypoints"].get(source)
        if landmark is None or keypoint is None:
            continue
        if not np.allclose(landmark, keypoint, atol=1e-8, rtol=0):
            raise ValueError(
                f"Saved direct landmark {name} disagrees with its source {source}"
            )
        targets[name] = LandmarkTarget(Point.from_array(values=landmark.copy()), scale)
    return targets


def build_data(
    path,
    *,
    first_frame=196,
    count=8,
    target_scale_mm=10.0,
    pose_scale_radians=0.5,
    angular_speed_scale=1.0,
    linear_speed_scale=1000.0,
    max_evaluations=100,
):
    if count < 2:
        raise ValueError("Select at least two frames")
    recorded, fit, provenance = read_recording(path, include_model=True)
    model = provenance.pop("model")
    snapshot = provenance.pop("skeleton")
    skeleton = SkeletonSnapshot.from_dict(snapshot).restore()
    rest_orientations = {
        name: RotationQuaternion(**value)
        for name, value in model["rest_pose"]["orientations"].items()
    }
    sources = target_sources(model)
    parents = {j.child.name: j.parent.name for j in skeleton.joints.values()}
    names = {skeleton.landmarks[n].segment for n in sources}
    for name in tuple(names):
        while name in parents:
            name = parents[name]
            names.add(name)
    names = frozenset(names)
    (root,) = names - set(parents)
    records = [r for r in recorded if first_frame <= r["number"] < first_frame + count]
    if len(records) != count or records[0]["number"] != first_frame:
        raise ValueError("Requested consecutive frame range is not available")
    inputs = []
    for record in records:
        if not names.issubset(record["rotations"]) or root not in record["origins"]:
            raise ValueError(
                f"Frame {record['number']} lacks saved initialization poses; choose another window"
            )
        world = {}
        for name in names:
            q = record["rotations"][name]
            if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
                raise ValueError(f"Saved rotation is not unit length: {name}")
            world[name] = RotationQuaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        local = {
            n: world[parents[n]].conjugate() * world[n] for n in names if n != root
        }
        inputs.append(
            ConnectedFrameInput(
                record["time"],
                observed_targets(record, sources, target_scale_mm),
                local,
                world[root],
                Point.from_array(values=record["origins"][root].copy()),
                pose_prior_orientations=rest_orientations,
            )
        )
    print(
        f"Fitting saved frames {first_frame}–{first_frame+count-1}; {len(names)} segments; data read-only",
        flush=True,
    )
    result = fit_connected_sequence(
        skeleton=skeleton,
        fit=fit,
        segment_names=names,
        frames=inputs,
        rotation_tolerances_radians=dict.fromkeys(names - {root}, pose_scale_radians),
        root_tolerances=RootPoseTolerances(1000.0, 1.0),
        motion_tolerances=MotionTolerances(
            dict.fromkeys(names - {root}, angular_speed_scale),
            angular_speed_scale,
            linear_speed_scale,
        ),
        max_evaluations=max_evaluations,
    )
    frames = []
    for record, input_frame, solved in zip(records, inputs, result.frames):
        connected = {
            n: dict(
                origin=solved.world_origins[n].array.tolist(),
                end=solved.landmarks[
                    skeleton.segments[n].frame_definition.primary_point_name
                ].array.tolist(),
                axes=[
                    solved.world_orientations[n].rotate_vector(vector=a).tolist()
                    for a in np.eye(3)
                ],
            )
            for n in sorted(names)
        }
        frames.append(
            dict(
                number=record["number"],
                time=record["time"],
                saved=saved_segments(record, snapshot, fit),
                connected=connected,
                points={n: p.tolist() for n, p in record["points"].items()},
                keypoints={n: p.tolist() for n, p in record["keypoints"].items()},
                status=dict(
                    converged=result.converged,
                    termination=result.termination,
                    evaluations=result.evaluations,
                    observed_targets=len(input_frame.targets),
                    missing_targets=sorted(set(sources) - set(input_frame.targets)),
                    connected_target_errors_mm=solved.target_errors,
                    lumbar_thoracic_bend_degrees=spine_bend_degrees(connected),
                    rigid_geometry_checks=rigid_target_checks(
                        skeleton, fit, input_frame.targets
                    ),
                ),
            )
        )
    if digest(Path(path)) != provenance["sha256"]:
        raise RuntimeError("Source recording changed during fitting")
    provenance.update(
        method="Experimental Forge sequence fit compared with saved production segments; neither is ground truth.",
        frame_range=[first_frame, first_frame + count - 1],
        target_sources=sources,
        initialization="Saved per-frame root and parent-relative rotations; no landmark remapping or dimension refit.",
        joint_pose_prior="Authored rest rotations from this recording's saved model; independent of the observed initial segment rotations.",
        root_pose_prior="Saved per-frame root; explicit weak translation/rotation preference, not independent measurement.",
        settings=dict(
            target_scale_mm=target_scale_mm,
            pose_scale_radians=pose_scale_radians,
            angular_velocity_scale_radians_per_second=angular_speed_scale,
            linear_velocity_scale_mm_per_second=linear_speed_scale,
            root_pose_scale_mm=1000.0,
            root_pose_scale_radians=1.0,
            max_evaluations=max_evaluations,
        ),
        warning="Explicit experimental weights, not calibrated uncertainty. Only the upper body is fitted; saved legs/hands remain available as context.",
        objective=dict(initial=result.initial_cost, final=result.final_cost),
    )
    print(
        f"Converged: {result.converged}; {result.evaluations} evaluations; {result.termination}",
        flush=True,
    )
    return dict(
        frames=frames,
        initial_index=min(4, count - 1),
        lengths=dict(fit.segment_lengths),
        segment_layers={"saved": 0xFFA860, "connected": 0x55AAFF},
        labels={
            "saved": "Saved posthoc segment",
            "connected": "Experimental connected segment",
        },
        provenance=provenance,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("test", "sample"), default="test")
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--first-frame", type=int, default=196)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--target-scale-mm", type=float, default=10.0)
    parser.add_argument("--pose-scale-radians", type=float, default=0.5)
    parser.add_argument("--angular-speed-scale", type=float, default=1.0)
    parser.add_argument("--linear-speed-scale", type=float, default=1000.0)
    parser.add_argument("--max-evaluations", type=int, default=100)
    args = vars(parser.parse_args())
    path = args.pop("parquet") or recording_path(args.pop("dataset"))
    args.pop("dataset", None)
    data = build_data(path, **args)
    folder = Path(__file__).resolve().parent
    html = folder.joinpath("recording_fit_viewer.html.template").read_text(
        encoding="utf-8"
    )
    html = html.replace(
        "__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    )
    html = html.replace("__ASSETS__", vendored_scripts() + geometry_script()).replace(
        "__VIEWER__",
        folder.joinpath("real_skeleton_viewer.js").read_text(encoding="utf-8"),
    )
    output = folder / "recording_fit_viewer.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
