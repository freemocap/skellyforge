"""Compare full reference-recording reads with a read-only Git source baseline."""
import argparse
import json
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from skellyforge.tools.recording_data import read_recording, recording_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="HEAD")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", f"{args.baseline}:skellyforge/tools/recording_data.py"],
        cwd=Path(__file__).resolve().parents[1], text=True, encoding="utf-8")
    namespace = {"__name__": "reference_reader_baseline"}
    exec(compile(source, "<reference reader baseline>", "exec"), namespace)
    measurements = []
    for dataset in ("test", "sample"):
        path = recording_path(dataset)
        times = {"before": [], "after": []}
        for repeat in range(4):
            results = {}
            for mode in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                start = perf_counter()
                results[mode] = (namespace["read_recording"] if mode == "before" else read_recording)(path, landmark_kind="MAPPED_KEYPOINTS_3D")
                elapsed = perf_counter() - start
                if repeat:
                    times[mode].append(elapsed)
            before, after = results["before"], results["after"]
            assert before[2] == after[2]
            assert before[1].segment_lengths == after[1].segment_lengths
            for a, b in zip(before[0], after[0], strict=True):
                assert a.keys() == b.keys()
                for key in a:
                    if isinstance(a[key], dict):
                        assert list(a[key]) == list(b[key])
                        for name in a[key]:
                            np.testing.assert_array_equal(a[key][name], b[key][name])
                    else:
                        assert a[key] == b[key]
        medians = {k: statistics.median(v) for k, v in times.items()}
        measurements.append(dict(dataset=dataset, frames=len(after[0]), median_seconds=medians,
            samples_seconds=times, exact_parity=True, speedup=medians["before"]/medians["after"], sha256=after[2]["sha256"]))
    print(json.dumps(dict(baseline=args.baseline, measurements=measurements), indent=2))


if __name__ == "__main__":
    main()
