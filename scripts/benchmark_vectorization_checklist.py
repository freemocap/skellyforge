"""Incremental benchmarks using SkellyForge's prepared reference-data loader.

Baseline functions come from Git without modifying the checkout. Shared helpers
use current implementations: timings isolate this checklist item, not all changes.
Run with --item landmarks or --item relative. JSON is written to stdout.
"""

import argparse
import importlib
import inspect
import json
import subprocess
import sys
import time
import types
from dataclasses import is_dataclass, fields
from pathlib import Path
from statistics import median

import numpy as np
from beartype import beartype

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_pose import (
    SkeletonPose,
    SegmentPose,
    PoseSolution,
)
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.tools.recording_data import read_recording, recording_path, digest

BASELINE = "0ede33e0a6e323dcd4f4b9de2e1b4fc9eff5e41d"


def baseline_module(name):
    if "_baseline_" + name in sys.modules:
        return sys.modules["_baseline_" + name]
    source = subprocess.check_output(
        ["git", "show", f'{BASELINE}:{name.replace(".", "/")}.py'],
        cwd=Path(__file__).resolve().parents[1],
        encoding="utf-8",
    )
    module = types.ModuleType("_baseline_" + name)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[module.__name__] = module
    exec(compile(source, name, "exec"), vars(module))
    for key, value in list(vars(module).items()):
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            setattr(module, key, beartype(value))
    return module


def flatten(value):
    if isinstance(value, dict):
        return {key: flatten(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [flatten(item) for item in value]
    if isinstance(value, RotationQuaternion):
        return value.as_array()
    if isinstance(value, Point):
        return value.array
    if hasattr(value, "landmarks"):
        return (
            flatten(value.segment_rotations_world),
            flatten(value.segment_origins),
            flatten(value.landmarks),
            value.inferred_segments,
        )
    if is_dataclass(value):
        return {
            field.name: flatten(getattr(value, field.name)) for field in fields(value)
        }
    return value


def compare(before, after):
    if isinstance(before, dict):
        assert before.keys() == after.keys()
        return max((compare(before[key], after[key]) for key in before), default=0.0)
    if isinstance(before, (tuple, list)):
        assert len(before) == len(after)
        return max(
            (compare(a, b) for a, b in zip(before, after, strict=True)), default=0.0
        )
    if isinstance(before, np.ndarray):
        if np.array_equal(after, before):
            return 0.0
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-10)
        return float(np.max(np.abs(after - before))) if before.size else 0.0
    if isinstance(before, float):
        assert abs(before - after) <= 1e-10
        return abs(before - after)
    assert before == after
    return 0.0


def benchmark(old, new, calls):
    times = {"before": [], "after": []}
    maximum = 0.0
    for repeat in range(4):
        outputs = {}
        for label in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
            function = old if label == "before" else new
            start = time.perf_counter()
            outputs[label] = [function(**kwargs) for kwargs in calls]
            elapsed = time.perf_counter() - start
            if repeat:
                times[label].append(elapsed)
        maximum = max(
            maximum, compare(flatten(outputs["before"]), flatten(outputs["after"]))
        )
    return dict(
        calls=len(calls),
        seconds=times,
        medians={k: median(v) for k, v in times.items()},
        max_absolute_difference=maximum,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        choices=[
            "landmarks",
            "relative",
            "angles",
            "hydration",
            "reductions",
            "inertia",
            "scale",
        ],
        required=True,
    )
    args = parser.parse_args()
    result = {}
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        records, fit, metadata = read_recording(
            path, landmark_kind="MAPPED_KEYPOINTS_3D"
        )
        skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
        rest = RestPose.default_for(skeleton=skeleton)
        poses = [
            SkeletonPose(
                {
                    name: SegmentPose(
                        segment_name=name,
                        origin=Point.from_array(
                            values=np.asarray(record["origins"][name])
                        ),
                        orientation=RotationQuaternion(*map(float, q)),
                        scale_estimate=fit.segment_scales[name],
                        solved_by=PoseSolution.RIGID_FIT,
                    )
                    for name, q in record["rotations"].items()
                }
            )
            for record in records
        ]
        measurements = {}
        if args.item == "landmarks":
            for module_name, function_name, calls in [
                (
                    "skellyforge.core.skeleton.pose.completion",
                    "complete_pose",
                    [
                        dict(
                            skeleton=skeleton,
                            rest=rest,
                            observed=pose,
                            segment_scales=fit.segment_scales,
                        )
                        for pose in poses
                        if rest.root_segment_name in pose.segment_poses
                    ],
                ),
                (
                    "skellyforge.core.biomechanics.center_of_mass",
                    "landmark_world_positions",
                    [
                        dict(
                            skeleton=skeleton,
                            pose=pose,
                            segment_scales=fit.segment_scales,
                        )
                        for pose in poses
                    ],
                ),
                (
                    "skellyforge.core.skeleton.pose.rest_pose",
                    "build_rest_pose",
                    [
                        dict(
                            skeleton=skeleton,
                            parents=rest.parents,
                            connect_ats=rest.connect_ats,
                            orientations=pose.parent_relative_orientations(
                                parents=rest.parents
                            ),
                            segment_scales=fit.segment_scales,
                        )
                        for pose in poses
                        if set(pose.segment_poses) == set(skeleton.segments)
                    ],
                ),
            ]:
                assert calls, function_name
                measurements[function_name] = benchmark(
                    getattr(baseline_module(module_name), function_name),
                    getattr(importlib.import_module(module_name), function_name),
                    calls,
                )
        elif args.item == "scale":
            from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton

            samples = {}
            for record in records:
                observed = {
                    key: Point(np.asarray(value))
                    for key, value in record["points"].items()
                }
                pose = hydrate_skeleton(
                    skeleton=skeleton, observed=observed, require_all=False
                )
                for key, segment in pose.segment_poses.items():
                    samples.setdefault(key, []).append(segment.scale_estimate)
            name = "skellyforge.core.skeleton.pose.model_scale_fitting"
            measurements["fit_model_scale_100_calls"] = benchmark(
                baseline_module(name).fit_model_scale,
                importlib.import_module(name).fit_model_scale,
                [
                    dict(
                        skeleton=skeleton,
                        scale_samples=samples,
                        voting_segment_names=fit.voting_segment_names,
                    )
                ]
                * 100,
            )
        elif args.item == "inertia":
            from skellyforge.core.biomechanics.center_of_mass import (
                CenterOfMassDefinitions,
            )
            from skellyforge.core.biomechanics.anthropometric_parameters import (
                AnthropometricParameters,
            )

            name = "skellyforge.core.biomechanics.composite_inertia"
            old, new = baseline_module(name), importlib.import_module(name)
            # Compare with the earlier reduction improvements already enabled.
            old.whole_body_center_of_mass = new.whole_body_center_of_mass
            old.whole_body_inertia_tensor = new.whole_body_inertia_tensor
            definitions = CenterOfMassDefinitions.from_default_yaml()
            parameters = AnthropometricParameters.from_default_yaml()
            measurements["body_inertial_properties"] = benchmark(
                old.body_inertial_properties,
                new.body_inertial_properties,
                [
                    dict(
                        skeleton=skeleton,
                        pose=pose,
                        body_mass=70.0,
                        anthropometric=parameters,
                        com_definitions=definitions,
                        segment_scales=fit.segment_scales,
                    )
                    for pose in poses
                    if rest.root_segment_name in pose.segment_poses
                ],
            )
        elif args.item == "reductions":
            from skellyforge.core.biomechanics.center_of_mass import (
                CenterOfMassDefinitions,
                landmark_world_positions,
                compute_segment_coms,
            )
            from skellyforge.core.biomechanics.anthropometric_parameters import (
                AnthropometricParameters,
            )
            from skellyforge.core.biomechanics.segment_inertia import (
                segment_inertia_tensor,
            )

            name = "skellyforge.core.biomechanics.composite_inertia"
            old, new = baseline_module(name), importlib.import_module(name)
            definitions = CenterOfMassDefinitions.from_default_yaml()
            parameters = AnthropometricParameters.from_default_yaml()
            com_calls, inertia_calls = [], []
            for pose in poses:
                world = landmark_world_positions(
                    skeleton=skeleton, pose=pose, segment_scales=fit.segment_scales
                )
                coms = compute_segment_coms(definitions=definitions, world=world)
                masses, inertias, available = {}, {}, {}
                for definition in definitions.definitions.values():
                    parameter = parameters.get(name=definition.name)
                    for full_name, side in definition.side_entries:
                        proximal = definition.resolve_landmark(
                            side=side, landmark=definition.proximal
                        )
                        distal = definition.resolve_landmark(
                            side=side, landmark=definition.distal
                        )
                        if (
                            proximal not in world
                            or distal not in world
                            or full_name not in coms
                        ):
                            continue
                        masses[full_name] = (
                            parameter.mass_fraction * 70.0
                        )  # fixed benchmark mass, not a measured subject mass
                        available[full_name] = coms[full_name]
                        inertias[full_name] = segment_inertia_tensor(
                            mass=masses[full_name],
                            radii=parameter.radii_of_gyration,
                            proximal=world[proximal],
                            distal=world[distal],
                        )
                com_calls.append(dict(segment_coms=available, segment_masses=masses))
                center = old.whole_body_center_of_mass(
                    segment_coms=available, segment_masses=masses
                )
                if center is not None:
                    inertia_calls.append(
                        dict(
                            segment_inertias=inertias,
                            segment_coms=available,
                            segment_masses=masses,
                            body_center_of_mass=center,
                        )
                    )
            measurements["whole_body_center_of_mass"] = benchmark(
                old.whole_body_center_of_mass, new.whole_body_center_of_mass, com_calls
            )
            measurements["whole_body_inertia_tensor"] = benchmark(
                old.whole_body_inertia_tensor,
                new.whole_body_inertia_tensor,
                inertia_calls,
            )
        elif args.item == "hydration":
            name = "skellyforge.core.skeleton.pose.hydration"
            measurements["hydrate_skeleton"] = benchmark(
                baseline_module(name).hydrate_skeleton,
                importlib.import_module(name).hydrate_skeleton,
                [
                    dict(
                        skeleton=skeleton,
                        require_all=False,
                        observed={
                            key: Point(np.asarray(value))
                            for key, value in record["points"].items()
                        },
                    )
                    for record in records
                ],
            )
        elif args.item == "angles":
            name = "skellyforge.core.skeleton.linkage.joint_pose"
            measurements["compute_joint_poses"] = benchmark(
                baseline_module(name).compute_joint_poses,
                importlib.import_module(name).compute_joint_poses,
                [dict(skeleton=skeleton, pose=pose) for pose in poses],
            )
        else:
            old = baseline_module(
                "skellyforge.core.skeleton.skeleton_pose"
            ).SkeletonPose.parent_relative_orientations
            old = beartype(old)
            measurements["parent_relative_orientations"] = benchmark(
                lambda pose: old(pose, parents=rest.parents),
                lambda pose: pose.parent_relative_orientations(parents=rest.parents),
                [dict(pose=pose) for pose in poses],
            )
        assert digest(path) == metadata["sha256"]
        result[dataset] = dict(
            frames=len(records), sha256=metadata["sha256"], measurements=measurements
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
