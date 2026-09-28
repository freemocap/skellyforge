"""Compare exact shared spine length with independent lengths on the aligned sample."""
import json
import argparse
from pathlib import Path
import numpy as np
from scripts.recording_data import recording_path, read_recording
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_rest_length_comparison import add_review_metrics
from scripts.solver_processing_comparison import publish
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows

OUTPUT = Path(__file__).resolve().parents[1] / 'build/shared_spine_length'
RATIOS = (20., 20., 12.)
REST_LENGTH_FRACTION = .5
FUNCTION_TOLERANCE = 1e-6
MAX_ITERATIONS = 200


def length_metrics(candidate):
    method = candidate['runs'][0]['methods']['full_body']
    lengths = np.array([[f['diagnostics'][name+' length (mm)']
                         for name in ('sacrolumbar', 'thoracic', 'cervical_spine')]
                        for f in method['frames']])
    total = lengths.sum(axis=1)
    result = {}
    for name, values in zip(('sacrolumbar', 'thoracic', 'cervical_spine', 'total'), [*lengths.T, total]):
        result[name] = dict(mean_mm=float(values.mean()), std_mm=float(values.std()),
            min_mm=float(values.min()), max_mm=float(values.max()),
            step_p95_mm=float(np.percentile(np.abs(np.diff(values)), 95)))
    result['maximum_proportion_error_mm'] = float(np.max(np.abs(lengths-total[:, None]*np.array(RATIOS)/sum(RATIOS))))
    # These measure variation, not error against anatomical ground truth.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    path = recording_path('sample')
    records, _, metadata = read_recording(path)
    baseline_path = OUTPUT.parent / 'sample_performance/tolerance_1e-06.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if baseline['metadata']['recording']['sha256'] != metadata['sha256']:
        raise ValueError('Regenerate the independent-length baseline for this prepared recording')
    def progress(window, total):
        if window['index'] % 100 == 0 or window['index'] == total-1:
            print(f"Window {window['index']+1}/{total}: {window['seconds']:.3f}s; converged={window['converged']}", flush=True)
    def solve(**kwargs):
        return fit_windows(kwargs, active_frames=3, max_iterations=MAX_ITERATIONS,
                           function_tolerance=FUNCTION_TOLERANCE, progress=progress)
    if args.publish_only:
        candidate = json.loads((OUTPUT/'shared.json').read_text(encoding='utf-8'))
    else:
        candidate = recording_body_catalog(path, 0, records[-1]['number'],
            free_axial_lengths=True, free_length_rest_prior=True, length_prior_fraction=REST_LENGTH_FRACTION,
            chest_line_prior=True, shoulder_profile='lower_closer_sc', relaxed_shoulders=True,
            proportional_spine_lengths=True, spine_proportion_ratios=RATIOS,
            shared_spine_length=True, solve_sequence=solve)
        add_context(candidate)
        add_review_metrics(candidate)
    if candidate['metadata']['recording']['sha256'] != metadata['sha256']:
        raise ValueError('Shared fit belongs to a different prepared recording')
    for value in (baseline, candidate):
        metrics = length_metrics(value)
        summary = value['runs'][0]['methods']['full_body']['summary']
        summary['Total spine length maximum (mm)'] = metrics['total']['max_mm']
        summary['Total spine length standard deviation (mm)'] = metrics['total']['std_mm']
        summary['Total spine length step p95 (mm)'] = metrics['total']['step_p95_mm']
        summary['Maximum deviation from exact spine proportions (mm)'] = metrics['maximum_proportion_error_mm']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'shared.json').write_text(json.dumps(candidate, allow_nan=False), encoding='utf-8')
    publish({'independent': baseline, 'shared': candidate}, comparison='shared_length')
    report = {name: dict(summary=value['runs'][0]['methods']['full_body']['summary'], length_variation=length_metrics(value),
                        unfinished_windows=[w for w in value['runs'][0]['methods']['full_body']['processing']['windows'] if not w['converged']])
              for name, value in [('independent', baseline), ('shared', candidate)]}
    (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
