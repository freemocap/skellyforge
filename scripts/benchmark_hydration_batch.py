"""Compare existing per-frame pose recovery against chunked reference-data recovery."""
import json
import statistics
from time import perf_counter

import numpy as np

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.hydration_batch import LandmarkPoseGeometry
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def run_batch(skeleton, observed):
    names = tuple(dict.fromkeys(n for frame in observed for n in frame))
    geometry = LandmarkPoseGeometry.from_skeleton(skeleton=skeleton, landmark_names=names)
    lookup = {n: i for i, n in enumerate(names)}
    output = []
    for begin in range(0, len(observed), 128):
        chunk = observed[begin:begin+128]
        positions = np.full((len(chunk), len(names), 3), np.nan)
        present = np.zeros(positions.shape[:2], dtype=bool)
        for frame, points in enumerate(chunk):
            columns = [lookup[n] for n in points]
            if columns:
                positions[frame, columns] = [p.array for p in points.values()]
                present[frame, columns] = True
        output.append(geometry.recover(positions=positions, landmark_present=present))
    return geometry, output


def main():
    measurements = []
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        records, _, metadata = read_recording(path, landmark_kind="MAPPED_KEYPOINTS_3D")
        skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
        observed = [{n: Point.from_array(values=np.asarray(v)) for n, v in r["points"].items()} for r in records]
        times = {"before": [], "after": []}
        maximum = dict(origins=0., orientations=0., scale_estimates=0.)
        for repeat in range(4):
            outputs = {}
            for mode in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                start = perf_counter()
                outputs[mode] = ([hydrate_skeleton(skeleton=skeleton, observed=p, require_all=False) for p in observed]
                    if mode == "before" else run_batch(skeleton, observed))
                duration = perf_counter() - start
                if repeat:
                    times[mode].append(duration)
            geometry, chunks = outputs["after"]
            arrays = {k: np.concatenate([getattr(c, k) for c in chunks]) for k in (*maximum, "solutions", "status")}
            for frame, pose in enumerate(outputs["before"]):
                assert set(pose.segment_poses) == {n for i, n in enumerate(geometry.segment_names) if arrays["status"][frame, i] == 1}
                for i, name in enumerate(geometry.segment_names):
                    if name not in pose.segment_poses:
                        continue
                    p = pose.segment_poses[name]
                    for key, expected in (("origins", p.origin.array), ("orientations", p.orientation.as_array()), ("scale_estimates", p.scale_estimate)):
                        actual = arrays[key][frame, i]
                        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
                        maximum[key] = max(maximum[key], float(np.max(np.abs(actual - expected))))
                    assert arrays["solutions"][frame, i] is p.solved_by
        assert digest(path) == metadata["sha256"]
        medians = {k: statistics.median(v) for k, v in times.items()}
        measurements.append(dict(dataset=dataset, frames=len(records), median_seconds=medians,
            samples_seconds=times, max_absolute_difference=maximum, speedup=medians["before"]/medians["after"], sha256=metadata["sha256"]))
    print(json.dumps(dict(chunk_size=128, includes="geometry setup and input packing", measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
