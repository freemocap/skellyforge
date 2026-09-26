"""Reduce SC forward depth while retaining the accepted 20:20:12 window fit."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.recording_data import recording_path, read_recording
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_processing_comparison import publish, RECORDING_FUNCTION_TOLERANCE
from scripts.solver_window_sequence import fit_windows

BUILD = Path(__file__).resolve().parent.parent / 'build'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    baseline = json.loads((BUILD/'spine_ratio_comparison/neck_12.json').read_text(encoding='utf-8'))
    folder = BUILD/'sc_offset_comparison'
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder/'forward_75.json'
    if args.publish_only:
        candidate = json.loads(cache.read_text(encoding='utf-8'))
    else:
        path = recording_path()
        records, _, _ = read_recording(path)
        def solve(**kwargs):
            def progress(window, total):
                if window['index'] % 40 == 0 or window['index'] == total-1:
                    print(f"Window {window['index']+1}/{total}", flush=True)
            return fit_windows(kwargs, active_frames=3,
                function_tolerance=RECORDING_FUNCTION_TOLERANCE, progress=progress)
        started = perf_counter()
        candidate = recording_body_catalog(path, 0, records[-1]['number'],
            free_axial_lengths=True, chest_line_prior=True, shoulder_profile='lower_closer_sc',
            relaxed_shoulders=True, proportional_spine_lengths=True,
            spine_proportion_ratios=(20.,20.,12.), solve_sequence=solve)
        candidate['runs'][0]['methods']['full_body']['summary']['This invocation read and fit wall seconds'] = perf_counter()-started
        add_context(candidate)
        cache.write_text(json.dumps(candidate, allow_nan=False), encoding='utf-8')
    publish({'neck_12': baseline, 'sc_forward_75': candidate}, comparison='shoulders')


if __name__ == '__main__':
    main()
