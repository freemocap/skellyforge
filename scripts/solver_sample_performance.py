"""Full sample recording: accepted fit, varying only Ceres stopping tolerance."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.recording_data import recording_path, read_recording
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_rest_length_comparison import add_review_metrics
from scripts.solver_processing_comparison import publish
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows

CASES = (('tolerance_1e-06', 1e-6, None), ('tolerance_1e-05', 1e-5, None),
         ('tolerance_0.0001', 1e-4, None), ('tolerance_warm_1e-05', 1e-5, 1e-6))
OUTPUT = Path(__file__).resolve().parent.parent / 'build/sample_performance'


def comparison_metrics(reference, candidate):
    a = reference['runs'][0]['methods']['full_body']
    b = candidate['runs'][0]['methods']['full_body']
    if reference['metadata']['recording']['sha256'] != candidate['metadata']['recording']['sha256']:
        raise ValueError('Different input recordings')
    distances = []
    angles = []
    per_frame = []
    for fa, fb in zip(a['frames'], b['frames'], strict=True):
        frame_distances = []
        for ba, bb in zip(fa['bodies'], fb['bodies'], strict=True):
            differences = np.linalg.norm(np.asarray(ba['fitted'])-np.asarray(bb['fitted']), axis=-1).tolist()
            distances.extend(differences)
            frame_distances.extend(differences)
            qa = Rotation.from_quat(ba['quaternion'], scalar_first=True)
            qb = Rotation.from_quat(bb['quaternion'], scalar_first=True)
            angles.append(float(np.degrees((qa.inv()*qb).magnitude())))
        per_frame.append({'frame': fa['diagnostics']['Recording frame'],
                          'maximum_landmark_difference_mm': max(frame_distances)})
    return {'fitted_landmark_difference_mm_p50_p95_max': np.percentile(distances,[50,95,100]).tolist(),
            'segment_quaternion_difference_degrees_p50_p95_max': np.percentile(angles,[50,95,100]).tolist(),
            'largest_difference_frames': sorted(per_frame, key=lambda f:f['maximum_landmark_difference_mm'], reverse=True)[:10]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = recording_path('sample')
    records, _, metadata = read_recording(path)
    print(f'Sample input: {len(records)} frames; timestamps {records[0]["time"]} to {records[-1]["time"]}', flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = {}
    report = {}
    for name, tolerance, initial_tolerance in CASES:
        cache = OUTPUT / (name+'.json')
        if args.publish_only or (args.resume and cache.exists()):
            candidate = json.loads(cache.read_text(encoding='utf-8'))
        else:
            def progress(window, total):
                if window['index'] % 100 == 0 or window['index']==total-1:
                    print(name, f"window {window['index']+1}/{total}",
                          f"{window['seconds']:.3f}s {window['iterations']} iterations", flush=True)
            def solve(**kwargs):
                return fit_windows(kwargs, active_frames=3, max_iterations=200,
                                   function_tolerance=tolerance, initial_function_tolerance=initial_tolerance, progress=progress)
            started = perf_counter()
            candidate = recording_body_catalog(path, 0, records[-1]['number'],
                free_axial_lengths=True, free_length_rest_prior=True,
                length_prior_fraction=.5, chest_line_prior=True,
                shoulder_profile='lower_closer_sc', relaxed_shoulders=True,
                proportional_spine_lengths=True, spine_proportion_ratios=(20.,20.,12.),
                solve_sequence=solve)
            method = candidate['runs'][0]['methods']['full_body']
            method['summary']['This invocation read and fit wall seconds'] = perf_counter()-started
            add_context(candidate)
            add_review_metrics(candidate)
            cache.write_text(json.dumps(candidate, allow_nan=False), encoding='utf-8')
        method = candidate['runs'][0]['methods']['full_body']
        if candidate['metadata']['recording']['sha256'] != metadata['sha256']:
            raise ValueError('Cached fit belongs to a different prepared recording')
        settings = method['settings']
        if (not settings['free_length_rest_prior'] or settings['length_prior_fraction'] != .5
                or settings['shoulder_geometry']['profile'] != 'lower_closer_sc'
                or list(settings['length_proportion_prior']['ratios']) != [20.,20.,12.]
                or len(method['frames']) != len(records)):
            raise ValueError('Cached fit differs from the accepted full-recording experiment')
        if method['processing']['function_tolerance'] != tolerance:
            raise ValueError('Cached tolerance differs')
        if method['processing'].get('initial_function_tolerance') != initial_tolerance:
            raise ValueError('Cached initial tolerance differs')
        candidates[name] = candidate
        iterations = [w['iterations'] for w in method['processing']['windows']]
        method['summary']['Window iterations p50'] = float(np.median(iterations))
        method['summary']['Window iterations p95'] = float(np.percentile(iterations,95))
        method['summary']['Window iterations max'] = max(iterations)
        report[name] = {'summary': method['summary'],
                        'difference_from_baseline': comparison_metrics(next(iter(candidates.values())), candidate)}
        (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        publish(candidates, comparison='performance')
        print('Saved', name, json.dumps(report[name]), flush=True)


if __name__ == '__main__':
    main()
