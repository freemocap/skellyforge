"""Display saved posthoc results without rerunning skeleton calculations."""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

if __package__:
    from .recording_data import recording_path, read_recording
    from .viewer_assets import vendored_scripts, geometry_script
else:
    from recording_data import recording_path, read_recording
    from viewer_assets import vendored_scripts, geometry_script

FOLDER = Path(__file__).resolve().parent


def saved_segments(record, skeleton, fit):
    """Convert recorded rigid transforms to drawing endpoints, without fitting."""
    landmarks = {p["name"]: p for p in skeleton["landmarks"]}
    segments = {s["name"]: s for s in skeleton["segments"]}
    if set(segments) != set(fit.segment_lengths) or set(segments) != set(
        fit.segment_scales
    ):
        raise ValueError("Saved skeleton and dimensions must cover the same segments")
    if (set(record["origins"]) | set(record["rotations"])) - set(segments):
        raise ValueError("Saved poses contain unknown segments")
    result = {}
    for name, segment in segments.items():
        if name not in record["origins"] or name not in record["rotations"]:
            continue
        q = record["rotations"][name]
        if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
            raise ValueError(f"Saved rotation is not a unit quaternion: {name}")
        w, x, y, z = q
        matrix = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        vector = (
            np.array(landmarks[segment["frame"]["primary_point"]]["position"])
            * fit.segment_scales[name]
        )
        if not np.isclose(
            np.linalg.norm(vector), fit.segment_lengths[name], rtol=1e-9, atol=1e-9
        ):
            raise ValueError(f"Saved dimensions disagree with saved geometry: {name}")
        origin = record["origins"][name]
        result[name] = dict(
            origin=origin.tolist(),
            end=(origin + matrix @ vector).tolist(),
            axes=matrix.T.tolist(),
            quaternion_wxyz=q.tolist(),
        )
    return result


def build_data(path, sensor_group=None):
    recorded, fit, provenance = read_recording(path, sensor_group)
    skeleton = provenance.pop("skeleton")
    frames = []
    for record in recorded:
        segments = saved_segments(record, skeleton, fit)
        frames.append(
            dict(
                number=record["number"],
                time=record["time"],
                points={n: p.tolist() for n, p in record["points"].items()},
                keypoints={n: p.tolist() for n, p in record["keypoints"].items()},
                saved=segments,
                status=dict(
                    saved_segments=len(segments),
                    missing_segments=len(fit.segment_lengths) - len(segments),
                ),
            )
        )
    provenance["fit_policy"] = (
        "Use the recorded fixed scale fit unchanged. Local default skeleton definitions are not used."
    )
    return dict(frames=frames, lengths=dict(fit.segment_lengths), provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("test", "sample"), default="test")
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--sensor-group")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    data = build_data(
        args.parquet or recording_path(args.dataset),
        args.sensor_group,
    )
    template = FOLDER.joinpath("real_skeleton_viewer.html.template").read_text(
        encoding="utf-8"
    )
    html = template.replace(
        "__ASSETS__", vendored_scripts() + "\n" + geometry_script()
    ).replace("__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c"))
    html = html.replace(
        "__VIEWER__",
        FOLDER.joinpath("real_skeleton_viewer.js").read_text(encoding="utf-8"),
    )
    output = FOLDER / "real_skeleton_viewer.html"
    output.write_text(html, encoding="utf-8")
    print(output)
    if args.serve:
        print(f"http://127.0.0.1:{args.port}/real_skeleton_viewer.html", flush=True)
        with ThreadingHTTPServer(
            ("127.0.0.1", args.port),
            partial(SimpleHTTPRequestHandler, directory=str(FOLDER)),
        ) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
