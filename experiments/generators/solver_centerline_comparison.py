"""Compare only chest-center distance strength on the full prepared sample."""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from skellyforge import _native
from skellyforge.tools.recording_data import recording_path, digest
from experiments.generators.solver_recording_body import recording_body_catalog
from experiments.generators.solver_recording_context import add_context
from experiments.generators.solver_rest_length_comparison import add_review_metrics
from experiments.generators.solver_axial_axes import add_axis_geometry
from experiments.generators.solver_spine_preferences import add_twist_diagnostics
from experiments.generators.solver_sc_anterior import add_sc_diagnostics
from experiments.generators.solver_spine_stability import metrics, BASE_REST_SCALE, TOTAL_LENGTH_BOUNDS, TWIST_SCALE
from experiments.generators.solver_shared_spine_length import RATIOS, FUNCTION_TOLERANCE, MAX_ITERATIONS
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows
from experiments.generators.solver_processing_comparison import publish

OUTPUT = REPO_ROOT / 'build/centerline'
DISTANCE_SCALES_MM = (10., 5.)


def summarize(candidate):
    method = candidate['runs'][0]['methods']['full_body']
    distances = [np.hypot(f['chest_line']['lateral_mm'], f['chest_line']['anterior_mm'])
                 for f in method['frames'] if f.get('chest_line') is not None]
    method['summary']['Chest center distance from line RMS (mm)'] = float(np.sqrt(np.mean(np.square(distances))))
    method['summary']['Chest center distance from line p95 (mm)'] = float(np.percentile(distances, 95))
    return metrics(candidate)


def main():
    prepare_output()
    path = recording_path('sample')
    baseline = json.loads((OUTPUT.parent/'sc_anterior/baseline.json').read_text())
    method = baseline['runs'][0]['methods']['full_body']
    identity = method['benchmark_case']
    if identity['source_sha'] != digest(path) or identity['native_sha'] != digest(Path(_native.__file__)):
        raise ValueError('Rerun the baseline for the current recording and native build first')
    start, end = [method['frames'][i]['diagnostics']['Recording frame'] for i in (0, -1)]
    method['comparison_label'] = 'Chest centerline / 50 mm'
    candidates = {'baseline': baseline}
    report = {'baseline': summarize(baseline)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for scale in DISTANCE_SCALES_MM:
        name = f'centerline_{scale:g}'
        def progress(window, total):
            if window['index'] % 200 == 0 or window['index'] == total-1:
                print(name, window['index']+1, '/', total, flush=True)
        def solve(**kwargs):
            return fit_windows(kwargs, active_frames=3, max_iterations=MAX_ITERATIONS,
                               function_tolerance=FUNCTION_TOLERANCE, progress=progress)
        started = perf_counter()
        candidate = recording_body_catalog(path, start, end, free_axial_lengths=True,
            free_length_rest_prior=True, length_prior_fraction=BASE_REST_SCALE,
            chest_line_prior=True, chest_line_distance_scale=scale,
            shoulder_profile='lower_closer_sc', relaxed_shoulders=True,
            proportional_spine_lengths=True, spine_proportion_ratios=RATIOS,
            shared_spine_length=True, shoulder_axis_preference=True,
            shared_total_bound_fractions=TOTAL_LENGTH_BOUNDS, spine_twist_scale=TWIST_SCALE,
            solve_sequence=solve)
        fit_wall = perf_counter()-started
        context_started = perf_counter()
        add_context(candidate); add_review_metrics(candidate); add_axis_geometry(candidate)
        add_twist_diagnostics(candidate); add_sc_diagnostics(candidate)
        method = candidate['runs'][0]['methods']['full_body']
        method['comparison_label'] = f'Chest centerline / {scale:g} mm'
        method['benchmark_case'] = dict(source_sha=digest(path), native_sha=digest(Path(_native.__file__)),
                                        chest_line_distance_scale_mm=scale)
        method['benchmark_timing'] = dict(ceres_seconds=method['frames'][0]['seconds'],
            window_processing_wall_seconds=method['processing']['wall_seconds'],
            read_fit_diagnostics_wall_seconds=fit_wall,
            viewer_context_wall_seconds=perf_counter()-context_started,
            total_before_serialization_wall_seconds=perf_counter()-started)
        report[name] = summarize(candidate)
        candidates[name] = candidate
        (OUTPUT/(name+'.json')).write_text(json.dumps(candidate, allow_nan=False))
        (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print('DONE', name, json.dumps(report[name]['summary']), flush=True)
    publish(candidates, comparison='centerline')


if __name__ == '__main__':
    main()
