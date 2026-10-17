"""Reference-recording comparison: per-frame positions versus bounded frame batches.

Run from the repository with its Python. Outputs JSON to stdout. Includes batch
geometry preparation and legacy input packing; excludes reading recordings and
parity comparison. This measures positions only, not full reconstruction.
"""

import argparse
import json
import statistics
import time

import numpy as np

from benchmark_synthesis import prepare
from skellyforge.core.skeleton.chain.anchored_batch import AnchoredSkeletonGeometry
from skellyforge.core.skeleton.chain.synthesis import synthesize_anchored_pose
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def run_batch(prepared, chunk_size):
    # This bridge is deliberately timed: current callers still produce objects.
    groups = {}
    for index, arguments in enumerate(prepared):
        groups.setdefault(arguments["anchor_segment_name"], []).append((index, arguments))
    output = []
    for anchor, entries in groups.items():
        geometry = AnchoredSkeletonGeometry.from_skeleton(skeleton=entries[0][1]["skeleton"],
            fit=entries[0][1]["fit"], anchor_segment_name=anchor)
        indices = {name: i for i, name in enumerate(geometry.segment_names)}
        for start in range(0, len(entries), chunk_size):
            chunk = entries[start:start + chunk_size]
            rotations = np.full((len(chunk), len(indices), 4), np.nan)
            present = np.zeros(rotations.shape[:2], dtype=bool)
            anchors = np.empty((len(chunk), 3))
            for frame, (_, arguments) in enumerate(chunk):
                names = list(arguments["segment_world_orientations"])
                columns = [indices[name] for name in names]
                rotations[frame, columns] = [arguments["segment_world_orientations"][n].as_array() for n in names]
                present[frame, columns] = True
                anchors[frame] = arguments["anchor_origin"].array
            result = geometry.positions(world_orientations=rotations, segment_present=present, anchor_origins=anchors)
            output.append(([i for i, _ in chunk], geometry, result))
    return output


def compare(before, after):
    count = 0
    for indices, geometry, result in after:
        for row, original_index in enumerate(indices):
            _, origins, landmarks = before[original_index]
            for names, mask, values, expected in [
                (geometry.segment_names, result.segment_present, result.segment_origins, origins),
                (geometry.landmark_names, result.landmark_present, result.landmark_positions, landmarks),
            ]:
                assert {n for i, n in enumerate(names) if mask[row, i]} == set(expected)
                for column, name in enumerate(names):
                    if name in expected:
                        np.testing.assert_array_equal(values[row, column], expected[name].array)
                        count += 3
                    else:
                        assert np.isnan(values[row, column]).all()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.repeats < 1:
        parser.error("chunk-size and repeats must be positive")
    measurements = []
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        records, fit, metadata = read_recording(path, landmark_kind="MAPPED_KEYPOINTS_3D")
        skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
        prepared = prepare(records, skeleton, fit)
        times = {"before": [], "after": []}
        for repeat in range(args.repeats + 1):
            results = {}
            for mode in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                started = time.perf_counter()
                results[mode] = ([synthesize_anchored_pose(**p) for p in prepared] if mode == "before"
                    else run_batch(prepared, args.chunk_size))
                elapsed = time.perf_counter() - started
                if repeat:
                    times[mode].append(elapsed)
            count = compare(results["before"], results["after"])
        assert digest(path) == metadata["sha256"]
        medians = {name: statistics.median(values) for name, values in times.items()}
        measurements.append(dict(dataset=dataset, motion_frames=len(records), connected_poses=len(prepared),
            compared_position_values=count, exact_parity=True, sha256=metadata["sha256"],
            samples_seconds=times, median_seconds=medians, speedup=medians["before"] / medians["after"],
            milliseconds_per_motion_frame={k: v * 1000 / len(records) for k, v in medians.items()}))
    print(json.dumps(dict(chunk_size=args.chunk_size, measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
