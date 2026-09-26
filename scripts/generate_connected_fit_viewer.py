"""Synthetic root/joint fitting comparison; not recorded production output.

Run from the Forge checkout: .venv/Scripts/python.exe -m scripts.generate_connected_fit_viewer
Then open scripts/connected_fit_viewer.html in a browser.
"""

import json
from pathlib import Path

import numpy as np

from scripts.viewer_assets import geometry_script, vendored_scripts
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.fit_connected_pose import (
    LandmarkTarget,
    RootPoseTolerances,
    fit_connected_pose,
)
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

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
"""Synthetic landmark observations only; no tracker adapter or tracker imports."""


def build_fixture():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    names = frozenset(
        (
            "pelvis",
            "sacrolumbar",
            "thoracic",
            "cervical_spine",
            "skull",
            "left_clavicle",
            "right_clavicle",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_arm",
            "right_lower_arm",
        )
    )
    fit = ModelScaleFit(
        fitted_scale=1700.0,
        segment_scales=dict.fromkeys(skeleton.segments, 1700.0),
        segment_lengths={n: s.length * 1700 for n, s in skeleton.segments.items()},
        measured_segment_names=frozenset(),
        voting_segment_names=frozenset(),
    )
    args = dict(
        skeleton=skeleton,
        fit=fit,
        segment_names=names,
        segment_relative_orientations={
            n: rest.relative_orientations[n] for n in names if n != "pelvis"
        },
        root_origin=Point.from_xyz(x=0.0, y=0.0, z=800.0),
        root_world_orientation=RotationQuaternion.identity(),
    )
    return args, rest


def build_data():
    args, rest = build_fixture()
    skeleton, fit, names = args["skeleton"], args["fit"], args["segment_names"]
    desired_args = {
        **args,
        "root_origin": Point.from_xyz(x=140.0, y=40.0, z=920.0),
        "root_world_orientation": RotationQuaternion.from_rotation_vector(
            rotation_vector=np.array([0.15, -0.2, 0.3])
        ),
    }
    raised = dict(args["segment_relative_orientations"])
    for side, sign in (("left", 1), ("right", -1)):
        parent_q = rest.segment_orientations[f"{side}_clavicle"]
        lift = RotationQuaternion.from_rotation_vector(
            rotation_vector=np.array([0.0, sign * 1.4, 0.0])
        )
        raised[f"{side}_upper_arm"] = (
            parent_q.conjugate() * lift * parent_q * raised[f"{side}_upper_arm"]
        )
    scenarios = [
        ("Whole-body displacement", desired_args, 0.0),
        ("Arms overhead", {**args, "segment_relative_orientations": raised}, 0.0),
    ]
    scenarios.extend(
        (f"Static T-pose, 1 mm noise, sample {i+1}", args, 1.0) for i in range(4)
    )
    rng = np.random.default_rng(42)
    frames = []
    cases = [
        ("Initial pose", None),
        ("Fit joints with pelvis fixed", False),
        ("Fit pelvis and joints together", True),
    ]

    def segments(world, origins, landmarks):
        return {
            n: dict(
                origin=origins[n].array.tolist(),
                end=landmarks[
                    skeleton.segments[n].frame_definition.primary_point_name
                ].array.tolist(),
                axes=[
                    world[n].rotate_vector(vector=axis).tolist() for axis in np.eye(3)
                ],
            )
            for n in sorted(names)
        }

    for scenario, desired_args, noise in scenarios:
        expected_world, expected_origins, desired = synthesize_fitted_pose(
            **desired_args
        )
        targets = {
            n: LandmarkTarget(
                Point.from_array(values=desired[n].array + rng.normal(0.0, noise, 3)),
                1.0,
            )
            for n in TARGET_NAMES
        }
        for label, free in (cases if noise == 0 else cases[-1:]):
            index = len(frames)
            status = {
                "scenario": scenario,
                "comparison": label,
                "noise_sd_mm_per_coordinate": noise,
            }
            if free is None:
                world, origins, landmarks = synthesize_fitted_pose(**args)
            else:
                result = fit_connected_pose(
                    **args,
                    targets=targets,
                    rotation_tolerances_radians=dict.fromkeys(
                        args["segment_relative_orientations"], 0.5
                    ),
                    root_tolerances=RootPoseTolerances(1000.0, 1.0) if free else None,
                )
                world, origins, landmarks = (
                    result.world_orientations,
                    result.world_origins,
                    result.landmarks,
                )
                status.update(
                    termination=result.termination,
                    iterations=result.iterations,
                    cost=result.final_cost,
                )
                status["maximum_local_rotation_error_degrees"] = max(
                    float(
                        np.rad2deg(
                            result.relative_orientations[n].angle_to(
                                other=desired_args["segment_relative_orientations"][n]
                            )
                        )
                    )
                    for n in args["segment_relative_orientations"]
                )
            status["target_errors_mm"] = {
                n: float(np.linalg.norm(landmarks[n].array - t.position.array))
                for n, t in targets.items()
            }
            frames.append(
                dict(
                    number=index,
                    time=float(index),
                    saved=segments(world, origins, landmarks),
                    reference=segments(expected_world, expected_origins, desired),
                    points={n: t.position.array.tolist() for n, t in targets.items()},
                    keypoints={},
                    status=status,
                )
            )
    return dict(
        frames=frames,
        initial_index=0,
        segment_layers={"saved": 0xFFA860, "reference": 0x55AAFF},
        lengths={n: fit.segment_lengths[n] for n in sorted(names)},
        labels={
            "saved": "Synthetic fitted segment",
            "reference": "Known synthetic segment",
            "points": "Synthetic target",
            "keypoints": "Unused",
        },
        provenance={
            "source": "Synthetic connected geometry; noise explicitly identified per case; RNG seed 42",
            "scope": "Pelvis, spine, head and arms; no fingers/legs or temporal solve",
            "target_tolerance_mm": 1.0,
            "joint_prior_radians": 0.5,
            "root_translation_prior_mm": 1000.0,
            "root_rotation_prior_radians": 1.0,
            "warning": "Demonstration settings, not calibrated production defaults",
        },
    )


def main():
    folder = Path(__file__).resolve().parent
    template = folder.joinpath("connected_fit_viewer.html.template").read_text(
        encoding="utf-8"
    )
    html = template.replace(
        "__DATA__", json.dumps(build_data(), allow_nan=False).replace("<", "\\u003c")
    )
    html = html.replace("__ASSETS__", vendored_scripts() + geometry_script())
    html = html.replace(
        "__VIEWER__",
        folder.joinpath("real_skeleton_viewer.js").read_text(encoding="utf-8"),
    )
    output = folder / "connected_fit_viewer.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
