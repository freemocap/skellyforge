"""Compare joint math on saved reference rotations; no input files are modified."""

import json
import statistics
from time import perf_counter

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.linkage.joint_batch import JointBatchGeometry
from skellyforge.core.skeleton.linkage.joint_pose import compute_joint_poses
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SegmentPose, SkeletonPose
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def run_batch(skeleton, poses):
    geometry = JointBatchGeometry.from_skeleton(skeleton=skeleton)
    lookup = {name: i for i, name in enumerate(geometry.segment_names)}
    output = []
    for begin in range(0, len(poses), 128):
        chunk = poses[begin:begin + 128]
        rotations = np.full((len(chunk), len(lookup), 4), np.nan)
        present = np.zeros(rotations.shape[:2], dtype=bool)
        solutions = np.full(present.shape, None, dtype=object)
        for frame, pose in enumerate(chunk):
            names = list(pose.segment_poses)
            columns = [lookup[name] for name in names]
            if columns:
                rotations[frame, columns] = [pose.segment_poses[n].orientation.as_array() for n in names]
                present[frame, columns] = True
                solutions[frame, columns] = [pose.segment_poses[n].solved_by for n in names]
        output.append(geometry.calculate(world_orientations=rotations, segment_present=present, segment_solutions=solutions))
    return geometry, output


def main():
    measurements = []
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        records, _, metadata = read_recording(path, landmark_kind="MAPPED_KEYPOINTS_3D")
        skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
        # Saved orientations supply the numerical inputs. DIRECTION is an explicit
        # benchmark-only provenance label, not a claim about these saved frames.
        poses = [SkeletonPose(segment_poses={name: SegmentPose(segment_name=name,
            origin=Point.from_xyz(x=0., y=0., z=0.), orientation=RotationQuaternion(*map(float, q)),
            scale_estimate=1., solved_by=PoseSolution.DIRECTION) for name, q in record["rotations"].items()})
            for record in records]
        timings = {"before": [], "after": []}
        maximum = 0.
        for repeat in range(4):
            outputs = {}
            for mode in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                start = perf_counter()
                outputs[mode] = ([compute_joint_poses(skeleton=skeleton, pose=p) for p in poses]
                    if mode == "before" else run_batch(skeleton, poses))
                duration = perf_counter() - start
                if repeat:
                    timings[mode].append(duration)
            geometry, chunks = outputs["after"]
            angles = np.concatenate([c.angles for c in chunks])
            rotations = np.concatenate([c.relative_orientations for c in chunks])
            present = np.concatenate([c.joint_present for c in chunks])
            count = 0
            for frame, original in enumerate(outputs["before"]):
                assert set(original) == {n for i, n in enumerate(geometry.joint_names) if present[frame, i]}
                for i, name in enumerate(geometry.joint_names):
                    if name in original:
                        expected = np.r_[original[name].relative_orientation.as_array(), original[name].angles]
                        actual = np.r_[rotations[frame, i], angles[frame, i]]
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
                        count += 7
        assert digest(path) == metadata["sha256"]
        medians = {k: statistics.median(v) for k, v in timings.items()}
        measurements.append(dict(dataset=dataset, motion_frames=len(records), compared_values=count,
            max_absolute_difference=maximum, median_seconds=medians, samples_seconds=timings,
            speedup=medians["before"] / medians["after"], sha256=metadata["sha256"]))
    print(json.dumps(dict(chunk_size=128, includes="geometry preparation and object-to-array input packing", measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
