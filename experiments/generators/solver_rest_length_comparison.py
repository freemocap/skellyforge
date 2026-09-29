"""Existing rest-length residual at two strengths, with both SC forward depths."""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation

from skellyforge.tools.recording_data import recording_path, read_recording
from experiments.generators.solver_recording_body import recording_body_catalog
from experiments.generators.solver_recording_context import add_context
from experiments.generators.solver_processing_comparison import publish, RECORDING_FUNCTION_TOLERANCE
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows

BUILD = REPO_ROOT / 'build'
REST_SCALES = (.5, .25)
PROFILES = {'100': 'lower_sc', '75': 'lower_closer_sc'}


def add_review_metrics(candidate):
    method = candidate['runs'][0]['methods']['full_body']
    neck = next(i for i, b in enumerate(candidate['bodies']) if b['id']=='cervical_spine')
    values = []
    for frame in method['frames']:
        line = frame.get('chest_line')
        if line is None:
            continue
        axis = Rotation.from_quat(frame['bodies'][neck]['quaternion'], scalar_first=True).apply([0,0,1])
        value = float(axis @ np.asarray(line['up']))
        frame['diagnostics']['Cervical axis dot hip-to-shoulder axis'] = value
        values.append(value)
    method['summary']['Frames with cervical axis opposite hip-to-shoulder axis'] = sum(v<0 for v in values)
    method['summary']['Minimum cervical axis dot hip-to-shoulder axis'] = min(values)


def main():
    prepare_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    candidates = {'neck_12': load(BUILD/'spine_ratio_comparison/neck_12.json'),
                  'sc_forward_75': load(BUILD/'sc_offset_comparison/forward_75.json')}
    for candidate in candidates.values():
        add_review_metrics(candidate)
    folder = BUILD/'rest_length_comparison'
    folder.mkdir(parents=True, exist_ok=True)
    path = recording_path()
    records, _, _ = read_recording(path)
    def solve(**kwargs):
        def progress(window, total):
            if window['index'] % 60 == 0 or window['index'] == total-1:
                print(f"Window {window['index']+1}/{total}", flush=True)
        return fit_windows(kwargs, active_frames=3,
            function_tolerance=RECORDING_FUNCTION_TOLERANCE, progress=progress)
    for depth, profile in PROFILES.items():
        for scale in REST_SCALES:
            name = f'sc_{depth}_rest_{int(scale*100)}'
            cache = folder/(name+'.json')
            if args.publish_only or (args.resume and cache.exists()):
                candidate = load(cache)
                settings = candidate['runs'][0]['methods']['full_body']['settings']
                if not settings['free_length_rest_prior'] or settings['length_prior_fraction']!=scale or settings['shoulder_geometry']['profile']!=profile:
                    raise ValueError('Cached case settings differ')
            else:
                print('Fitting', name, flush=True)
                started = perf_counter()
                candidate = recording_body_catalog(path, 0, records[-1]['number'],
                    free_axial_lengths=True, free_length_rest_prior=True,
                    length_prior_fraction=scale, chest_line_prior=True,
                    shoulder_profile=profile, relaxed_shoulders=True,
                    proportional_spine_lengths=True, spine_proportion_ratios=(20.,20.,12.),
                    solve_sequence=solve)
                candidate['runs'][0]['methods']['full_body']['summary']['This invocation read and fit wall seconds'] = perf_counter()-started
                add_context(candidate)
                cache.write_text(json.dumps(candidate, allow_nan=False), encoding='utf-8')
            add_review_metrics(candidate)
            candidates[name] = candidate
            publish(candidates, comparison='rest_lengths')
            print('Saved', name, flush=True)


if __name__ == '__main__':
    main()
