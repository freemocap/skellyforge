"""Compare reconstruction math with a Git revision on existing reference recordings.

Run from this repository: .venv/Scripts/python.exe -B scripts/benchmark_synthesis.py
No inputs are modified. JSON goes to stdout; redirect into .test-artifacts/.
The baseline uses its original synthesis functions and scalar vector rotation;
all other code, data, and runtime validation are shared with the candidate.
"""

import argparse
import ast
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from beartype import beartype

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_anchored_pose
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def original_functions(revision):
    root = Path(__file__).resolve().parents[1]

    def source(path):
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root, text=True, encoding="utf-8"
        )

    namespace = {"__name__": "__baseline_synthesis__"}
    exec(
        compile(
            source("skellyforge/core/skeleton/chain/synthesis.py"),
            "<baseline synthesis>",
            "exec",
        ),
        namespace,
    )
    for name, value in list(namespace.items()):
        if inspect.isfunction(value) and value.__module__ == namespace["__name__"]:
            namespace[name] = beartype(value)
    tree = ast.parse(source("skellyforge/core/math/geometry/rotation_quaternion.py"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RotationQuaternion"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "rotate_vector"
    )
    module = ast.Module(body=[method], type_ignores=[])
    from skellyforge.core.math.geometry import rotation_quaternion

    globals_copy = dict(vars(rotation_quaternion))
    exec(
        compile(ast.fix_missing_locations(module), "<baseline rotation>", "exec"),
        globals_copy,
    )
    return namespace["synthesize_anchored_pose"], beartype(
        globals_copy["rotate_vector"]
    )


def prepare(records, skeleton, fit):
    # Saved orientations can contain multiple measured components. Exercise each
    # component independently; never fill missing segments or invent an anchor.
    prepared = []
    for record in records:
        rotations = {
            name: RotationQuaternion(*map(float, q))
            for name, q in record["rotations"].items()
        }
        remaining = set(rotations)
        adjacency = {name: set() for name in remaining}
        for joint in skeleton.joints.values():
            parent, child = joint.parent.name, joint.child.name
            if parent in remaining and child in remaining:
                adjacency[parent].add(child)
                adjacency[child].add(parent)
        while remaining:
            anchor = min(remaining)
            component, pending = set(), [anchor]
            while pending:
                name = pending.pop()
                if name in component:
                    continue
                component.add(name)
                pending.extend(sorted(adjacency[name] - component))
            remaining -= component
            prepared.append(
                dict(
                    skeleton=skeleton,
                    fit=fit,
                    segment_world_orientations={
                        name: rotations[name] for name in sorted(component)
                    },
                    anchor_segment_name=anchor,
                    anchor_origin=Point.from_array(
                        values=np.asarray(record["origins"][anchor])
                    ),
                )
            )
    return prepared


def run(function, prepared):
    start = time.perf_counter()
    outputs = [function(**arguments) for arguments in prepared]
    elapsed = time.perf_counter() - start
    values = []
    keys = []
    for rotations, origins, landmarks in outputs:
        for kind, items in [
            ("rotations", rotations),
            ("origins", origins),
            ("landmarks", landmarks),
        ]:
            keys.append((kind, tuple(sorted(items))))
            for name in sorted(items):
                item = items[name]
                values.extend(item.as_array() if kind == "rotations" else item.array)
    return elapsed, keys, np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="HEAD")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    old_synthesis, old_rotation = original_functions(args.baseline)
    current_rotation = RotationQuaternion.rotate_vector
    results = []
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        records, fit, metadata = read_recording(
            path, landmark_kind="MAPPED_KEYPOINTS_3D"
        )
        skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
        prepared = prepare(records, skeleton, fit)
        assert prepared, "No saved poses to benchmark"
        times = {"before": [], "after": []}
        baseline_keys = baseline_values = None
        maximum_difference = 0.0
        try:
            for repeat in range(args.repeats + 1):
                # First pair warms both paths, then alternate execution order.
                for mode in (
                    ("before", "after") if repeat % 2 == 0 else ("after", "before")
                ):
                    RotationQuaternion.rotate_vector = (
                        old_rotation if mode == "before" else current_rotation
                    )
                    elapsed, keys, values = run(
                        old_synthesis if mode == "before" else synthesize_anchored_pose,
                        prepared,
                    )
                    if baseline_values is None:
                        baseline_keys, baseline_values = keys, values
                    assert keys == baseline_keys
                    np.testing.assert_allclose(
                        values, baseline_values, rtol=0, atol=1e-10
                    )
                    maximum_difference = max(
                        maximum_difference,
                        float(np.max(np.abs(values - baseline_values))),
                    )
                    if repeat:
                        times[mode].append(elapsed)
        finally:
            RotationQuaternion.rotate_vector = current_rotation
        assert digest(path) == metadata["sha256"], "Reference input changed"
        medians = {name: statistics.median(samples) for name, samples in times.items()}
        results.append(
            dict(
                dataset=dataset,
                path=str(path),
                sha256=metadata["sha256"],
                motion_frames=len(records),
                connected_poses=len(prepared),
                numeric_values=len(baseline_values),
                max_absolute_difference=maximum_difference,
                seconds=times,
                median_seconds=medians,
                milliseconds_per_motion_frame={
                    name: seconds * 1000 / len(records)
                    for name, seconds in medians.items()
                },
                speedup=medians["before"] / medians["after"],
            )
        )
    print(json.dumps(dict(baseline=args.baseline, measurements=results), indent=2))


if __name__ == "__main__":
    main()
