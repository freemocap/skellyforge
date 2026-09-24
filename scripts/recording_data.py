"""Read prepared recording landmarks as data, without importing the producer."""

import hashlib
import json
from pathlib import Path

import numpy as np

from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit


def recording_path(dataset="test", home=None):
    name = {"test": "freemocap_test_data", "sample": "freemocap_sample_data"}[dataset]
    root = (Path.home() if home is None else Path(home)) / "freemocap_data"
    candidates = [
        root
        / "testing"
        / "prepared"
        / name
        / "current"
        / "recordings"
        / name
        / f"{name}_data.parquet",
        root / "recordings" / name / f"{name}_data.parquet",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Processed recording missing. Prepare it in FreeMoCap first, or pass --parquet. Looked in:\n"
        + "\n".join(map(str, candidates))
    )


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_recording(path, sensor_group=None):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Real-data viewing requires PyArrow: install the recording-viewer extra."
        ) from error
    path = Path(path).resolve()
    before = digest(path)
    with pq.ParquetFile(path) as parquet:
        raw = (parquet.schema_arrow.metadata or {}).get(b"freemocap.recording")
        if raw is None:
            raise ValueError("Missing recording descriptor")
        metadata = json.loads(raw)
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported recording schema version")
        run_id = metadata["selected_run_id"]
        run = metadata["runs"][str(run_id)]
        channels = [
            c
            for c in run["channels"]
            if c["kind"] == "LANDMARKS_3D"
            and c["source"] == "model:standard_human"
            and (sensor_group is None or c["sensor_group"] == sensor_group)
        ]
        if len(channels) != 1:
            raise ValueError(
                "Select one standard-human landmark channel with --sensor-group"
            )
        channel = channels[0]
        if channel["components"] != {"x": "mm", "y": "mm", "z": "mm"}:
            raise ValueError("Expected XYZ landmarks in millimeters")
        reference = run["reference_frames"][channel["reference_frame"]]
        if reference.get("basis") != "blender_x_right_y_forward_z_up":
            raise ValueError(
                "Unsupported coordinate basis; no implicit axis conversion"
            )
        fits = [
            f
            for f in run["scale_fits"]
            if all(
                f[k] == channel[k]
                for k in ("source", "sensor_group", "reference_frame")
            )
        ]
        if len(fits) != 1 or fits[0]["units"] != "mm":
            raise ValueError("Recording needs one compatible prepared scale fit")
        saved_fit = fits[0]["fit"]
        fit = ModelScaleFit(
            **{
                **saved_fit,
                "measured_segment_names": frozenset(
                    saved_fit["measured_segment_names"]
                ),
                "voting_segment_names": frozenset(saved_fit["voting_segment_names"]),
            }
        )
        definition = run["sources"][channel["source"]]["definition"]
        keypoint_channels = [
            c
            for c in run["channels"]
            if c["source"] == definition["tracker"]
            and c["kind"] == definition["point_kind"]
            and c["sensor_group"] == channel["sensor_group"]
            and c["reference_frame"] == channel["reference_frame"]
        ]
        if (
            len(keypoint_channels) != 1
            or keypoint_channels[0]["components"] != channel["components"]
        ):
            raise ValueError("Expected one matching saved XYZ keypoint input channel")
        keypoint_channel = keypoint_channels[0]
        selected_channels = {
            channel["kind"]: channel,
            keypoint_channel["kind"]: keypoint_channel,
        }
        for kind, components in (
            ("SEGMENT_ORIGINS", channel["components"]),
            ("ROTATIONS_WORLD", {c: "1" for c in "wxyz"}),
        ):
            matches = [
                c
                for c in run["channels"]
                if c["kind"] == kind
                and all(
                    c[k] == channel[k]
                    for k in ("source", "sensor_group", "reference_frame")
                )
            ]
            if len(matches) != 1 or matches[0]["components"] != components:
                raise ValueError(f"Expected one compatible saved {kind} channel")
            selected_channels[kind] = matches[0]
        rows = {kind: [] for kind in selected_channels}
        for batch in parquet.iter_batches(batch_size=65536):
            columns = batch.to_pydict()
            for i, kind in enumerate(columns["channel"]):
                selected = selected_channels.get(kind)
                if selected is None or columns["run_id"][i] != run_id:
                    continue
                if any(
                    columns[k][i] != selected[k]
                    for k in ("source", "sensor_group", "reference_frame")
                ):
                    continue
                rows[kind].append(
                    {
                        k: columns[k][i]
                        for k in (
                            "frame_number",
                            "timestamp_s",
                            "name",
                            "component",
                            "value",
                            "units",
                        )
                    }
                )
    frames = decode_rows(rows[channel["kind"]], channel["names"])
    keypoints = decode_rows(rows[keypoint_channel["kind"]], keypoint_channel["names"])
    attach_keypoints(frames, keypoints)
    for kind, field, components, units in (
        ("SEGMENT_ORIGINS", "origins", "xyz", "mm"),
        ("ROTATIONS_WORLD", "rotations", "wxyz", "1"),
    ):
        values = decode_rows(
            rows[kind], selected_channels[kind]["names"], components, units
        )
        attach_series(frames, values, field)
    if digest(path) != before:
        raise RuntimeError("Recording changed while reading; regenerate the review")
    return (
        frames,
        fit,
        dict(
            path=str(path),
            sha256=before,
            run_id=run_id,
            channel=channel,
            keypoint_channel=keypoint_channel,
            skeleton=run["models"]["standard_human"]["skeleton"],
            method="Replay of saved segment origins, world rotations, fixed dimensions, landmarks and keypoints. No reconstruction, refitting or optimization in the viewer.",
        ),
    )


def attach_keypoints(frames, keypoints):
    """Join distinct point namespaces only on the identical saved frame grid."""
    attach_series(frames, keypoints, "keypoints")


def attach_series(frames, keypoints, field):
    if [(f["number"], f["time"]) for f in frames] != [
        (f["number"], f["time"]) for f in keypoints
    ]:
        raise ValueError(
            "Saved channels must have identical frame numbers and timestamps"
        )
    for frame, points in zip(frames, keypoints, strict=True):
        frame[field] = points["points"]


def decode_rows(rows, names, components="xyz", units="mm"):
    frames = {}
    seen = set()
    names = set(names)
    for row in rows:
        frame = row["frame_number"]
        time = row["timestamp_s"]
        name = row["name"]
        component = row["component"]
        key = (frame, name, component)
        if (
            name not in names
            or component not in components
            or row["units"] != units
            or key in seen
        ):
            raise ValueError("Unexpected or duplicate landmark sample")
        if frame < 0 or not np.isfinite(time):
            raise ValueError("Invalid frame or timestamp")
        seen.add(key)
        entry = frames.setdefault(frame, dict(number=frame, time=time, points={}))
        if entry["time"] != time:
            raise ValueError("Inconsistent timestamps within a frame")
        point = entry["points"].setdefault(name, np.full(len(components), np.nan))
        value = row["value"]
        if value is not None:
            if not np.isfinite(value):
                raise ValueError("Nonfinite landmark value must be null")
            point[components.index(component)] = value
    result = [frames[n] for n in sorted(frames)]
    if not result or not np.all(np.diff([f["time"] for f in result]) > 0):
        raise ValueError("Expected nonempty, strictly increasing frame timestamps")
    for frame in result:
        frame["points"] = {
            n: p for n, p in frame["points"].items() if np.isfinite(p).all()
        }
    return result
