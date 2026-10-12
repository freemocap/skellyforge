"""Reference-data timings for batched COM and measurement residuals."""
import json
import statistics
from time import perf_counter

import numpy as np

from skellyforge.core.biomechanics.center_of_mass import CenterOfMassDefinitions, compute_segment_coms
from skellyforge.core.biomechanics.center_of_mass_batch import CenterOfMassGeometry
from skellyforge.core.biomechanics.composite_inertia import whole_body_center_of_mass
from skellyforge.core.skeleton.pose.rigid_body_diagnostics import ResidualBatchGeometry, measure_rigid_body_residuals
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.model_scale_fitting import StreamingModelScaleFitter, fit_model_scale_arrays
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import digest, read_recording, recording_path


def main():
    measurements = []
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        records, fit, metadata = read_recording(path, landmark_kind="MAPPED_KEYPOINTS_3D")
        skeleton = SkeletonSnapshot.from_dict(metadata["skeleton"]).restore()
        worlds = [{n: np.asarray(v) for n, v in r["points"].items()} for r in records]
        poses = [hydrate_skeleton(skeleton=skeleton, observed={n: Point(v) for n, v in w.items()}, require_all=False) for w in worlds]
        definitions = CenterOfMassDefinitions.from_default_yaml()
        # Fixed illustrative masses; numerical comparison, not measured subject mass.
        masses = {n: 1. for d in definitions.definitions.values() for n, _ in d.side_entries}
        names = tuple(dict.fromkeys(n for w in worlds for n in w))
        segments = tuple(skeleton.segments)
        def before_com():
            return [whole_body_center_of_mass(segment_coms=compute_segment_coms(definitions=definitions, world=w), segment_masses=masses) for w in worlds]
        def after_com():
            geometry = CenterOfMassGeometry.from_definitions(definitions=definitions, landmark_names=names, segment_masses=masses)
            lookup = {n: i for i, n in enumerate(names)}
            result = []
            for begin in range(0, len(worlds), 128):
                chunk = worlds[begin:begin+128]
                values = np.full((len(chunk), len(names), 3), np.nan)
                mask = np.zeros(values.shape[:2], bool)
                for frame, world in enumerate(chunk):
                    columns = [lookup[n] for n in world]
                    if columns:
                        values[frame, columns] = list(world.values())
                        mask[frame, columns] = True
                result.append(geometry.calculate(positions=values, landmark_present=mask).body_positions)
            return np.concatenate(result)
        def before_residual():
            return [measure_rigid_body_residuals(skeleton=skeleton, pose=p, measured_segment_names=fit.voting_segment_names,
                reference_lengths=fit.segment_lengths, reference_kind="recording_fit") for p in poses]
        def after_residual():
            geometry = ResidualBatchGeometry.from_skeleton(skeleton=skeleton, measured_segment_names=fit.voting_segment_names,
                reference_lengths=fit.segment_lengths, reference_kind="recording_fit")
            lookup = {n: i for i, n in enumerate(segments)}
            result = []
            for begin in range(0, len(poses), 128):
                chunk = poses[begin:begin+128]
                scales = np.full((len(chunk), len(segments)), np.nan)
                mask = np.zeros(scales.shape, bool)
                for frame, pose in enumerate(chunk):
                    columns = [lookup[n] for n in pose.segment_poses]
                    scales[frame, columns] = [p.scale_estimate for p in pose.segment_poses.values()]
                    mask[frame, columns] = True
                result.append(geometry.calculate(scales=scales, segment_present=mask)[1])
            return np.concatenate(result)
        def before_fit():
            fitter = StreamingModelScaleFitter(skeleton=skeleton, voting_segment_names=fit.voting_segment_names, window_frames=len(poses))
            for pose in poses:
                fitter.observe_pose(pose=pose)
            return fitter.current_fit()
        def after_fit():
            lookup = {n: i for i, n in enumerate(segments)}
            scales = np.full((len(poses), len(segments)), np.nan)
            mask = np.zeros(scales.shape, bool)
            for frame, pose in enumerate(poses):
                columns = [lookup[n] for n in pose.segment_poses]
                scales[frame, columns] = [p.scale_estimate for p in pose.segment_poses.values()]
                mask[frame, columns] = True
            return fit_model_scale_arrays(skeleton=skeleton, segment_names=segments, scales=scales, segment_present=mask,
                voting_segment_names=fit.voting_segment_names)
        native_scales = np.full((len(poses), len(segments)), np.nan)
        native_present = np.zeros(native_scales.shape, bool)
        for frame, pose in enumerate(poses):
            for i, name in enumerate(segments):
                if name in pose.segment_poses:
                    native_scales[frame, i] = pose.segment_poses[name].scale_estimate
                    native_present[frame, i] = True
        def native_fit():
            return fit_model_scale_arrays(skeleton=skeleton, segment_names=segments, scales=native_scales,
                segment_present=native_present, voting_segment_names=fit.voting_segment_names)
        for label, before, after in (("center_of_mass", before_com, after_com), ("residuals", before_residual, after_residual),
                                    ("recording_scale_fit", before_fit, after_fit), ("recording_scale_fit_arrays_already_available", before_fit, native_fit)):
            times = {"before": [], "after": []}
            for repeat in range(4):
                output = {}
                for mode in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                    start = perf_counter()
                    output[mode] = (before if mode == "before" else after)()
                    if repeat:
                        times[mode].append(perf_counter() - start)
                if label.startswith("recording_scale_fit"):
                    assert output["before"].segment_scales == output["after"].segment_scales
                    assert output["before"].segment_lengths == output["after"].segment_lengths
                    assert output["before"].fitted_scale == output["after"].fitted_scale
                    assert output["before"].measured_segment_names == output["after"].measured_segment_names
                else:
                    expected = (np.asarray([np.full(3, np.nan) if p is None else p for p in output["before"]]) if label == "center_of_mass"
                        else np.asarray([[np.nan if row[n].residual is None else row[n].residual for n in segments] for row in output["before"]]))
                    np.testing.assert_array_equal(output["after"], expected)
            medians = {k: statistics.median(v) for k, v in times.items()}
            measurements.append(dict(dataset=dataset, operation=label, frames=len(records), median_seconds=medians,
                input_packing_included=label != "recording_scale_fit_arrays_already_available",
                samples_seconds=times, exact_parity=True, speedup=medians["before"]/medians["after"]))
        assert digest(path) == metadata["sha256"]
    print(json.dumps(dict(chunk_size=128, includes="geometry setup and input packing", measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
