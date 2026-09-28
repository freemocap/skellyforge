"""Compare explicitly selected spine proportions with identical three-frame solves."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.recording_data import recording_path, read_recording
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_processing_comparison import publish, RECORDING_FUNCTION_TOLERANCE
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows

RATIO_CASES = {'neck_12': (20., 20., 12.), 'neck_13': (20., 20., 13.),
               'neck_14': (20., 20., 14.)}
BUILD = Path(__file__).resolve().parent.parent / 'build'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse completed cached cases and fit the remaining cases')
    args = parser.parse_args()
    folder = BUILD / 'spine_ratio_comparison'
    folder.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((BUILD / 'processing_comparison/full/windows_3.json').read_text(encoding='utf-8'))
    candidates = {'windows_3': baseline}
    path = recording_path()
    records, _, _ = read_recording(path)

    def solve(**kwargs):
        def progress(window, total):
            if window['index'] % 40 == 0 or window['index'] == total-1:
                print(f"Window {window['index']+1}/{total}", flush=True)
        return fit_windows(kwargs, active_frames=3,
                           function_tolerance=RECORDING_FUNCTION_TOLERANCE, progress=progress)

    for name, ratios in RATIO_CASES.items():
        cache = folder / (name + '.json')
        if args.publish_only or (args.resume and cache.exists()):
            candidate = json.loads(cache.read_text(encoding='utf-8'))
            if candidate['runs'][0]['methods']['full_body']['settings']['length_proportion_prior']['ratios'] != list(ratios):
                raise ValueError('Cached ratios differ from requested case')
        else:
            print('Fitting', name, ratios, flush=True)
            started = perf_counter()
            candidate = recording_body_catalog(path, 0, records[-1]['number'],
                free_axial_lengths=True, chest_line_prior=True, shoulder_profile='lower_sc',
                relaxed_shoulders=True, proportional_spine_lengths=True,
                spine_proportion_ratios=ratios, solve_sequence=solve)
            method = candidate['runs'][0]['methods']['full_body']
            method['summary']['This invocation read and fit wall seconds'] = perf_counter()-started
            add_context(candidate)
            cache.write_text(json.dumps(candidate, allow_nan=False), encoding='utf-8')
        candidates[name] = candidate
        publish(candidates, comparison='ratios')
        print('Saved', name, flush=True)


if __name__ == '__main__':
    main()
