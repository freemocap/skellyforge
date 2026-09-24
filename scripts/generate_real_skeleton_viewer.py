"""Build a standalone real-recording skeleton viewer using only local Forge."""

import argparse
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.model_scale_fitting import fit_model_scale
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    fit_connected_pose,
    LandmarkTarget,
)
from skellyforge.core.math.geometry.spatial_vectors import Point

if __package__:
    from .recording_data import recording_path, read_recording
    from .viewer_assets import vendored_scripts, geometry_script
else:
    from recording_data import recording_path, read_recording
    from viewer_assets import vendored_scripts, geometry_script

FOLDER = Path(__file__).resolve().parent


def build_data(
    path, sensor_group=None, position_tolerance=5.0, rotation_tolerance_deg=30.0
):
    if (
        not np.isfinite([position_tolerance, rotation_tolerance_deg]).all()
        or min(position_tolerance, rotation_tolerance_deg) <= 0
    ):
        raise ValueError("Fitting tolerances must be finite and positive")
    recorded, saved_fit, provenance = read_recording(path, sensor_group)
    skeleton = SkeletonDefinition.from_default_yaml()
    # Saved fits may have been produced by older segment definitions. Re-estimate
    # dimensions once from the prepared landmarks, using current hydration.
    poses = [
        hydrate_skeleton(
            skeleton=skeleton,
            observed={
                n: Point.from_array(values=p) for n, p in record["points"].items()
            },
            require_all=False,
        )
        for record in recorded
    ]
    samples = {
        n: [
            pose.segment_poses[n].scale_estimate
            for pose in poses
            if n in pose.segment_poses
        ]
        for n in skeleton.segments
    }
    fit = fit_model_scale(
        skeleton=skeleton,
        scale_samples=samples,
        voting_segment_names=saved_fit.voting_segment_names,
    )
    provenance["saved_segment_lengths_mm"] = dict(saved_fit.segment_lengths)
    provenance["current_segment_lengths_mm"] = dict(fit.segment_lengths)
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    resolver = ContinuousRollResolver.for_skeleton(
        skeleton=skeleton, rest_relative_orientations=rest.relative_orientations
    )
    root = rest.root_segment_name
    frames = []
    upper = frozenset(
        ("pelvis", "sacrolumbar", "thoracic", "left_clavicle", "right_clavicle")
    )
    for index, record in enumerate(recorded):
        observed = {n: Point.from_array(values=p) for n, p in record["points"].items()}
        pose = resolver.resolve_pose(pose=poses[index])
        selected = set()

        def include(n):
            if n not in pose.segment_poses:
                return False
            parent = rest.parents[n]
            if parent is not None and not include(parent):
                return False
            selected.add(n)
            return True

        for n in pose.segment_poses:
            include(n)
        connected = {}
        fitted = {}
        status = {"termination": "unavailable"}
        if root in selected:
            local = pose.parent_relative_orientations(parents=rest.parents)
            local = {n: local[n] for n in selected if n != root}
            args = dict(
                skeleton=skeleton,
                fit=fit,
                segment_relative_orientations=local,
                root_origin=pose.segment_poses[root].origin,
                root_world_orientation=pose.segment_poses[root].orientation,
                segment_names=frozenset(selected),
            )
            world, origins, _ = synthesize_fitted_pose(**args)
            connected = serialize(skeleton, fit, world, origins)
            available = upper & selected
            targets = {
                n: LandmarkTarget(position=observed[n], tolerance=position_tolerance)
                for n in ("left_acromion", "right_acromion")
                if n in observed and skeleton.landmarks[n].segment in available
            }
            if targets and {"pelvis", "sacrolumbar", "thoracic"}.issubset(available):
                result = fit_connected_pose(
                    **{
                        **args,
                        "segment_names": frozenset(available),
                        "segment_relative_orientations": {
                            n: local[n] for n in available if n != root
                        },
                    },
                    targets=targets,
                    rotation_tolerances_radians={
                        n: float(np.deg2rad(rotation_tolerance_deg))
                        for n in available
                        if n != root
                    },
                )
                corrected = combine_fitted_world_rotations(
                    parents=rest.parents,
                    reference_world={
                        n: pose.segment_poses[n].orientation for n in selected
                    },
                    fitted_world=result.world_orientations,
                )
                world, origins, _ = synthesize_fitted_pose(
                    **{**args, "segment_relative_orientations": corrected}
                )
                fitted = serialize(skeleton, fit, world, origins)
                status = dict(
                    termination=result.termination,
                    iterations=result.iterations,
                    shoulder_errors_mm=result.target_errors,
                    targets_within_tolerance=result.targets_within_tolerance,
                    initial_cost=result.initial_cost,
                    final_cost=result.final_cost,
                )
        independent = serialize(
            skeleton,
            fit,
            {n: p.orientation for n, p in pose.segment_poses.items()},
            {n: p.origin for n, p in pose.segment_poses.items()},
        )
        frames.append(
            dict(
                number=record["number"],
                time=record["time"],
                points={n: p.tolist() for n, p in record["points"].items()},
                independent=independent,
                connected=connected,
                fitted=fitted,
                status=status,
            )
        )
        if index % 25 == 0:
            print(f"Prepared {index+1}/{len(recorded)} frames", flush=True)
    provenance["fit_policy"] = (
        "Refit fixed dimensions once using current Forge hydration of saved landmarks; "
        "reuse the saved scale-voting segment selection. No tracker remapping or video processing."
    )
    provenance["position_tolerance_mm"] = position_tolerance
    provenance["rotation_tolerance_deg"] = rotation_tolerance_deg
    import skellyforge.core.skeleton.pose.hydration as hydration
    import skellyforge.core.skeleton.pose.fit_connected_pose as fitter

    provenance["local_sources"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(hydration.__file__), Path(fitter.__file__), Path(__file__))
    }
    return dict(frames=frames, lengths=dict(fit.segment_lengths), provenance=provenance)


def combine_fitted_world_rotations(*, parents, reference_world, fitted_world):
    """Move fitted branches while preserving unfitted segments' world rotations.

    Descendant positions still follow the connected tree. Their relative rotations
    must compensate for fitted parent rotations; retaining old locals rotates them
    a second time even though those segments were not part of the fitting problem.
    """
    world = {**reference_world, **fitted_world}
    return {
        n: world[parents[n]].inverse() * q
        for n, q in world.items()
        if parents[n] is not None
    }


def serialize(skeleton, fit, rotations, origins):
    result = {}
    for n, q in rotations.items():
        origin = origins[n].array
        primary = skeleton.segments[n].frame_definition.primary_point_name
        vector = (
            skeleton.landmarks[primary].local_position.array * fit.segment_scales[n]
        )
        result[n] = dict(
            origin=origin.tolist(),
            end=(origin + q.rotate_vector(vector=vector)).tolist(),
            axes=q.to_rotation_matrix().T.tolist(),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("test", "sample"), default="test")
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--sensor-group")
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=30.0)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    data = build_data(
        args.parquet or recording_path(args.dataset),
        args.sensor_group,
        args.position_tolerance_mm,
        args.rotation_tolerance_deg,
    )
    template = FOLDER.joinpath("real_skeleton_viewer.html.template").read_text(
        encoding="utf-8"
    )
    html = template.replace(
        "__ASSETS__", vendored_scripts() + "\n" + geometry_script()
    ).replace("__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c"))
    html = html.replace(
        "__VIEWER__",
        FOLDER.joinpath("real_skeleton_viewer.js").read_text(encoding="utf-8"),
    )
    output = FOLDER / "real_skeleton_viewer.html"
    output.write_text(html, encoding="utf-8")
    print(output)
    if args.serve:
        print(f"http://127.0.0.1:{args.port}/real_skeleton_viewer.html", flush=True)
        with ThreadingHTTPServer(
            ("127.0.0.1", args.port),
            partial(SimpleHTTPRequestHandler, directory=str(FOLDER)),
        ) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
