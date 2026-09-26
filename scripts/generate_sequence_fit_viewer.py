"""Compare independent and time-window fits on identical synthetic observations."""

import json
from pathlib import Path
import numpy as np

from scripts.generate_connected_fit_viewer import build_fixture, TARGET_NAMES
from scripts.viewer_assets import geometry_script, vendored_scripts
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    LandmarkTarget,
    RootPoseTolerances,
    fit_connected_pose,
)
from skellyforge.core.skeleton.pose.fit_connected_sequence import (
    ConnectedFrameInput,
    MotionTolerances,
    fit_connected_sequence,
)


def build_data():
    args, rest = build_fixture()
    skeleton, fit, names = args["skeleton"], args["fit"], args["segment_names"]
    prior = dict.fromkeys(args["segment_relative_orientations"], 0.5)
    root_prior = RootPoseTolerances(1000.0, 1.0)
    motion = MotionTolerances(dict.fromkeys(prior, 1.0), 1.0, 1000.0)
    rng = np.random.default_rng(42)
    frames, cases = [], []

    def draw_geometry(world, origins, landmarks):
        return {
            n: dict(
                origin=origins[n].array.tolist(),
                end=landmarks[
                    skeleton.segments[n].frame_definition.primary_point_name
                ].array.tolist(),
                axes=[world[n].rotate_vector(vector=a).tolist() for a in np.eye(3)],
            )
            for n in sorted(names)
        }

    for moving in (False, True):
        label = (
            "Moving arms and pelvis + 1 mm noise"
            if moving
            else "Static T-pose + 1 mm noise"
        )
        inputs, truth, independent = [], [], []
        for i in range(4):
            local = dict(args["segment_relative_orientations"])
            if moving:
                for side, sign in (("left", 1), ("right", -1)):
                    parent = rest.segment_orientations[f"{side}_clavicle"]
                    turn = RotationQuaternion.from_rotation_vector(
                        rotation_vector=np.array([0.0, sign * 0.12 * i, 0.0])
                    )
                    local[f"{side}_upper_arm"] = (
                        parent.conjugate() * turn * parent * local[f"{side}_upper_arm"]
                    )
            desired = {
                **args,
                "segment_relative_orientations": local,
                "root_origin": Point.from_array(
                    values=args["root_origin"].array
                    + [5.0 * i if moving else 0.0, 0.0, 0.0]
                ),
            }
            world, origins, landmarks = synthesize_fitted_pose(**desired)
            truth.append((world, origins, landmarks))
            targets = {
                n: LandmarkTarget(
                    Point.from_array(
                        values=landmarks[n].array + rng.normal(0.0, 1.0, 3)
                    ),
                    1.0,
                )
                for n in TARGET_NAMES
            }
            inputs.append(
                ConnectedFrameInput(
                    i / 30.0,
                    targets,
                    args["segment_relative_orientations"],
                    args["root_world_orientation"],
                    args["root_origin"],
                )
            )
            independent.append(
                fit_connected_pose(
                    **args,
                    targets=targets,
                    rotation_tolerances_radians=prior,
                    root_tolerances=root_prior,
                )
            )
        result = fit_connected_sequence(
            skeleton=skeleton,
            fit=fit,
            segment_names=names,
            frames=inputs,
            rotation_tolerances_radians=prior,
            root_tolerances=root_prior,
            motion_tolerances=motion,
        )
        print(
            label, result.converged, result.evaluations, result.termination, flush=True
        )
        cases.append(
            dict(label=label, first=len(frames), last=len(frames) + len(inputs) - 1)
        )
        for i, (input_frame, single, together, known) in enumerate(
            zip(inputs, independent, result.frames, truth)
        ):
            frames.append(
                dict(
                    number=i,
                    time=(10.0 if moving else 0.0) + input_frame.time_seconds,
                    points={
                        n: t.position.array.tolist()
                        for n, t in input_frame.targets.items()
                    },
                    keypoints={},
                    saved=draw_geometry(
                        single.world_orientations,
                        single.world_origins,
                        single.landmarks,
                    ),
                    sequence=draw_geometry(
                        together.world_orientations,
                        together.world_origins,
                        together.landmarks,
                    ),
                    reference=draw_geometry(*known),
                    status=dict(
                        case=label,
                        independent_termination=single.termination,
                        sequence_converged=result.converged,
                        sequence_termination=result.termination,
                        sequence_evaluations=result.evaluations,
                        independent_errors_mm=single.target_errors,
                        sequence_errors_mm=together.target_errors,
                    ),
                )
            )
    return dict(
        frames=frames,
        cases=cases,
        initial_index=0,
        lengths={n: fit.segment_lengths[n] for n in sorted(names)},
        segment_layers={"saved": 0xFFA860, "sequence": 0x66EE99, "reference": 0x55AAFF},
        labels={
            "saved": "Independent fit",
            "sequence": "Sequence fit",
            "reference": "Known synthetic pose",
            "points": "Noisy synthetic observation",
        },
        provenance=dict(
            source="Synthetic data only; 30 Hz, 4-frame windows; XYZ Gaussian noise SD 1 mm; seed 42",
            initialization="Every frame starts at the same T-pose. Known motion is not supplied as a pose prior.",
            target_tolerance_mm=1.0,
            joint_pose_prior_radians=0.5,
            root_pose_prior_mm=1000.0,
            root_pose_prior_radians=1.0,
            joint_velocity_scale_radians_per_second=1.0,
            root_velocity_scale_radians_per_second=1.0,
            root_velocity_scale_mm_per_second=1000.0,
            warning="Demonstration settings, not production defaults. Velocity regularization can attenuate real motion; inspect the moving case.",
        ),
    )


def main():
    folder = Path(__file__).resolve().parent
    data = build_data()
    template = folder.joinpath("sequence_fit_viewer.html.template").read_text(
        encoding="utf-8"
    )
    html = template.replace(
        "__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    )
    html = html.replace("__ASSETS__", vendored_scripts() + geometry_script()).replace(
        "__VIEWER__",
        folder.joinpath("real_skeleton_viewer.js").read_text(encoding="utf-8"),
    )
    output = folder / "sequence_fit_viewer.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
