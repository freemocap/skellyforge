"""Full-sample SC half-space comparison; exact cervical attachment retained."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from skellyforge import _native
from scripts.recording_data import recording_path, digest, read_recording
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_rest_length_comparison import add_review_metrics
from scripts.solver_axial_axes import add_axis_geometry
from scripts.solver_spine_preferences import add_twist_diagnostics
from scripts.solver_sc_anterior import SC_ANTERIOR_SCALE_MM, add_sc_diagnostics
from scripts.solver_spine_stability import metrics, BASE_REST_SCALE, TOTAL_LENGTH_BOUNDS, TWIST_SCALE
from scripts.solver_shared_spine_length import RATIOS, FUNCTION_TOLERANCE, MAX_ITERATIONS
from scripts.solver_window_sequence import fit_windows
from scripts.solver_processing_comparison import publish

OUTPUT=Path(__file__).resolve().parents[1]/'build/sc_anterior'
CASES=(('baseline','Bounded length + twist',None),
       ('sc_anterior','Bounded + twist + SC anterior',SC_ANTERIOR_SCALE_MM))


def write_report(report):
    lines=['# SC anterior comparison', '',
           'Same full sample, 1108 frames at 30 Hz. Only the SC anterior residual changes.', '',
           '| Metric | Baseline | SC anterior |', '| --- | ---: | ---: |']
    keys=['Observed landmark RMS (mm)','SC posterior violation RMS (mm)',
          'SC posterior violation maximum (mm)','SC posterior frames',
          'SC anterior distance standard deviation (mm)',
          'Neck anterior distance standard deviation (mm)',
          'Cervical attachment error maximum (mm)','Relative spine twist RMS (degrees)',
          'Windows not converged']
    for key in keys:
        lines.append('| '+key+' | '+' | '.join(f"{v['summary'][key]:.4f}" for v in report.values())+' |')
    for key in next(iter(report.values()))['timing']:
        lines.append('| '+key+' | '+' | '.join(f"{v['timing'][key]:.3f}" for v in report.values())+' |')
    lines+=['', 'Timing: single sequential runs. Ceres time is included in window wall time; do not add them. Build, serialization and publication excluded.',
            'Whole-recording distance variation includes true movement; it is not a jitter or accuracy metric.', '',
            '| Residual family cost | Baseline | SC anterior |','| --- | ---: | ---: |']
    for family in next(iter(report.values()))['residual_costs']:
        values=[v['residual_costs'][family] for v in report.values()]
        lines.append('| '+family+' | '+' | '.join(f'{sum(v) if isinstance(v,list) else v:.4f}' for v in values)+' |')
    lines+=['','Costs are Ceres half-squared weighted sums, evaluated once on the assembled trajectory. Different objectives are not accuracy rankings.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only',action='store_true')
    args=parser.parse_args()
    path=recording_path('sample');source_sha=digest(path);native_sha=digest(Path(_native.__file__))
    records, _, recording = read_recording(path)
    if recording['sha256'] != source_sha:raise ValueError('Prepared recording changed while reading')
    start, end = records[0]['number'], records[-1]['number']
    del records
    OUTPUT.mkdir(parents=True,exist_ok=True);candidates={};report={}
    for name,label,scale in CASES:
        cache=OUTPUT/(name+'.json')
        spec=dict(name=name,sc_anterior_scale_mm=scale,source_sha=source_sha,native_sha=native_sha)
        if args.publish_only:
            candidate=json.loads(cache.read_text(encoding='utf-8'))
            if candidate['runs'][0]['methods']['full_body']['benchmark_case']!=spec:raise ValueError('Cached configuration differs')
        else:
            def progress(w,n):
                if w['index']%200==0 or w['index']==n-1:print(f"{name}: {w['index']+1}/{n}",flush=True)
            def solve(**kwargs):
                return fit_windows(kwargs,active_frames=3,max_iterations=MAX_ITERATIONS,function_tolerance=FUNCTION_TOLERANCE,progress=progress)
            print('START '+name,flush=True);started=perf_counter()
            candidate=recording_body_catalog(path,start,end,free_axial_lengths=True,free_length_rest_prior=True,
                length_prior_fraction=BASE_REST_SCALE,chest_line_prior=True,shoulder_profile='lower_closer_sc',relaxed_shoulders=True,
                proportional_spine_lengths=True,spine_proportion_ratios=RATIOS,shared_spine_length=True,
                shoulder_axis_preference=True,shared_total_bound_fractions=TOTAL_LENGTH_BOUNDS,spine_twist_scale=TWIST_SCALE,
                sc_anterior_scale=scale,solve_sequence=solve)
            fit_wall=perf_counter()-started;context_start=perf_counter()
            add_context(candidate);add_review_metrics(candidate);add_axis_geometry(candidate);add_twist_diagnostics(candidate);add_sc_diagnostics(candidate)
            method=candidate['runs'][0]['methods']['full_body'];method['benchmark_case']=spec
            method['benchmark_timing']=dict(ceres_seconds=method['frames'][0]['seconds'],window_processing_wall_seconds=method['processing']['wall_seconds'],
                read_fit_diagnostics_wall_seconds=fit_wall,viewer_context_wall_seconds=perf_counter()-context_start,
                total_before_serialization_wall_seconds=perf_counter()-started)
        method=candidate['runs'][0]['methods']['full_body'];method['comparison_label']=label
        if scale is not None:
            note=' One-sided SC anterior half-space preference; no anterior attraction or margin.'
            if note not in method['objective']:method['objective']+=note
        report[name]=metrics(candidate);candidates[name]=candidate
        cache.write_text(json.dumps(candidate,allow_nan=False),encoding='utf-8')
        (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('DONE '+name+' '+json.dumps(report[name]['summary']),flush=True)
    publish(candidates,comparison='sc_anterior')
    write_report(report)


if __name__=='__main__':main()
