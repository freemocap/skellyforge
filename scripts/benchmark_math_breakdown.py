"""Attribute reconstruction savings by enabling changes one at a time.

Run from SkellyForge; reads the existing prepared test and sample recordings.
Prints JSON. Does not change source data or the installed application.
"""

import json
import statistics
import time

import numpy as np

from benchmark_synthesis import original_functions, prepare, run
from skellyforge.core.math.geometry.rotation_quaternion import (
    RotationQuaternion,
    rotate_vector_pairs,
)
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain import synthesis
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def measure(operation):
    samples = []
    result = None
    for iteration in range(4):
        start = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - start
        if iteration:
            samples.append(elapsed)
    return dict(seconds=samples, median_seconds=statistics.median(samples)), result


def main():
    revision = "0ede33e0a6e323dcd4f4b9de2e1b4fc9eff5e41d"
    old_synthesis, old_rotation = original_functions(revision)
    old_globals = old_synthesis.__wrapped__.__globals__
    old_validation = old_globals["_validate_fitted_geometry"]
    new_rotation = RotationQuaternion.rotate_vector
    result = dict(baseline=revision, recordings=[])
    try:
        rng = np.random.default_rng(824)
        q = rng.normal(size=(10000, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        vectors = rng.normal(size=(10000, 3))
        rotations = [RotationQuaternion(*map(float, row)) for row in q]
        micro = {}
        outputs = []
        for label, function in [
            ("original_scalar", old_rotation),
            ("new_scalar", new_rotation),
        ]:
            micro[label], values = measure(
                lambda: np.asarray(
                    [
                        function(rotation, vector=vector)
                        for rotation, vector in zip(rotations, vectors, strict=True)
                    ]
                )
            )
            outputs.append(values)
        micro["paired_array"], values = measure(
            lambda: rotate_vector_pairs(quaternions=q, vectors=vectors)
        )
        outputs.append(values)
        for values in outputs[1:]:
            np.testing.assert_array_equal(values, outputs[0])
        result["rotation_10000_vectors"] = micro
        for dataset in ("test", "sample"):
            path = recording_path(dataset)
            records, fit, metadata = read_recording(
                path, landmark_kind="MAPPED_KEYPOINTS_3D"
            )
            skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
            prepared = prepare(records, skeleton, fit)
            timings = {
                name: []
                for name in (
                    "original",
                    "scalar_only",
                    "scalar_and_validation",
                    "all_changes",
                )
            }
            expected = expected_keys = None
            max_difference = 0.0
            for iteration in range(4):
                modes = list(timings)
                if iteration % 2:
                    modes.reverse()
                for mode in modes:
                    RotationQuaternion.rotate_vector = (
                        old_rotation if mode == "original" else new_rotation
                    )
                    old_globals["_validate_fitted_geometry"] = (
                        synthesis._validate_fitted_geometry
                        if mode == "scalar_and_validation"
                        else old_validation
                    )
                    elapsed, keys, values = run(
                        (
                            synthesis.synthesize_anchored_pose
                            if mode == "all_changes"
                            else old_synthesis
                        ),
                        prepared,
                    )
                    if expected is None:
                        expected, expected_keys = values, keys
                    assert keys == expected_keys
                    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-10)
                    max_difference = max(
                        max_difference, float(np.max(np.abs(values - expected)))
                    )
                    if iteration:
                        timings[mode].append(elapsed)
            validation = {}
            for label, function in [
                ("before", old_validation),
                ("after", synthesis._validate_fitted_geometry),
            ]:
                validation[label], _ = measure(
                    lambda: [function(skeleton=skeleton, fit=fit) for _ in records]
                )
            # Recompute measured segment orientations from mapped keypoints. This
            # exercises the scalar optimization beyond saved-rotation synthesis.
            observations = [
                {
                    name: Point(np.asarray(value))
                    for name, value in record["points"].items()
                }
                for record in records
            ]
            hydration = {}
            expected_pose_values = None
            expected_pose_keys = None
            for label, rotation in [("before", old_rotation), ("after", new_rotation)]:
                RotationQuaternion.rotate_vector = rotation
                hydration[label], poses = measure(
                    lambda: [
                        hydrate_skeleton(
                            skeleton=skeleton, observed=points, require_all=False
                        )
                        for points in observations
                    ]
                )
                keys = [tuple(sorted(pose.segment_poses)) for pose in poses]
                values = np.asarray(
                    [
                        value
                        for pose in poses
                        for name in sorted(pose.segment_poses)
                        for value in (
                            *pose.segment_poses[name].origin.array,
                            *pose.segment_poses[name].orientation.as_array(),
                            pose.segment_poses[name].scale_estimate,
                        )
                    ]
                )
                if expected_pose_values is None:
                    expected_pose_values, expected_pose_keys = values, keys
                assert keys == expected_pose_keys
                np.testing.assert_allclose(
                    values, expected_pose_values, rtol=0, atol=1e-10
                )
            assert digest(path) == metadata["sha256"]
            result["recordings"].append(
                dict(
                    dataset=dataset,
                    frames=len(records),
                    sha256=metadata["sha256"],
                    synthesis_seconds=timings,
                    synthesis_medians={
                        k: statistics.median(v) for k, v in timings.items()
                    },
                    synthesis_max_absolute_difference=max_difference,
                    validation=validation,
                    hydration=hydration,
                    hydration_numeric_values=len(values),
                    hydration_max_absolute_difference=float(
                        np.max(np.abs(values - expected_pose_values))
                    ),
                )
            )
    finally:
        RotationQuaternion.rotate_vector = new_rotation
        old_globals["_validate_fitted_geometry"] = old_validation
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
