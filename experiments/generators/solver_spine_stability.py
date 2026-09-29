"""Controlled real-sample comparisons of total length and relative spine twist."""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import argparse
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
from experiments.generators.solver_processing_comparison import publish
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows
from experiments.generators.solver_shared_spine_length import RATIOS, FUNCTION_TOLERANCE, MAX_ITERATIONS, length_metrics

OUTPUT=REPO_ROOT/'build/spine_stability'
BASE_REST_SCALE=.5
STRONG_REST_SCALE=.25
TOTAL_LENGTH_BOUNDS=(.6,1.25)
TWIST_SCALE=.5
CASES=(
    ('baseline','Baseline',BASE_REST_SCALE,None,None),
    ('strong_length','Stronger length',STRONG_REST_SCALE,None,None),
    ('bounded_length','Bounded length',BASE_REST_SCALE,TOTAL_LENGTH_BOUNDS,None),
    ('twist','Twist preference',BASE_REST_SCALE,None,TWIST_SCALE),
    ('strong_twist','Stronger length + twist',STRONG_REST_SCALE,None,TWIST_SCALE),
    ('bounded_twist','Bounded length + twist',BASE_REST_SCALE,TOTAL_LENGTH_BOUNDS,TWIST_SCALE),
)


def metrics(candidate):
    method=candidate['runs'][0]['methods']['full_body']
    costs=method['settings']['costs_by_family']
    cost_total=sum(sum(v) if isinstance(v,list) else v for v in costs.values())
    if not np.isclose(cost_total,method['frames'][0]['costs'][-1],rtol=1e-9,atol=1e-8):
        raise ValueError('Residual family costs do not sum to the evaluated full-trajectory objective')
    variation=length_metrics(candidate)
    summary=method['summary']
    summary['Total spine length maximum (mm)']=variation['total']['max_mm']
    summary['Total spine length standard deviation (mm)']=variation['total']['std_mm']
    bounds=method['settings'].get('shared_axial_length') or {}
    totals=[sum(f['lengths']) for f in method['frames']]
    summary['Frames at shared total length bounds']=sum(any(abs(t-bounds[k])<1e-5 for k in ('minimum_total_mm','maximum_total_mm') if k in bounds) for t in totals)
    windows=method['processing']['windows']
    summary['Window iteration p95']=float(np.percentile([w['iterations'] for w in windows],95))
    return dict(summary=summary,timing=method['benchmark_timing'],residual_costs=costs,
                total_objective_cost=cost_total,length_variation=variation,
                unfinished_windows=[{k:w[k] for k in ('active_start','active_end','iterations','initial_cost','final_cost','report')} for w in windows if not w['converged']],
                comparison_note='Residual costs include their respective scales; total costs across changed objectives are not an accuracy ranking.')


def write_report(report):
    lines=['# Full-sample spine comparison', '',
           'All options use the same prepared recording, 1108 frames, three active frames, shared 20:20:12 lengths and the SC/shoulder axis preference.', '',
           '| Option | Ceres s | Window wall s | Target RMS mm | Relative twist RMS deg | Maximum total length mm | Bound frames | Unconverged windows |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,item in report.items():
        s=item['summary'];t=item['timing']
        lines.append(f"| {name} | {t['ceres_seconds']:.2f} | {t['window_processing_wall_seconds']:.2f} | {s['Observed landmark RMS (mm)']:.3f} | {s['Relative spine twist RMS (degrees)']:.2f} | {s['Total spine length maximum (mm)']:.2f} | {s['Frames at shared total length bounds']} | {s['Windows not converged']} |")
    lines += ['', 'Target RMS uses the original mapped keypoint targets. Relative twist and length statistics describe the fitted model, not anatomical ground truth.', '',
              '## Wall-time breakdown', '', '| Option | Read + fit + diagnostics s | Viewer context s | Total before serialization s |', '| --- | ---: | ---: | ---: |']
    for name,item in report.items():
        t=item['timing'];lines.append(f"| {name} | {t['read_fit_diagnostics_wall_seconds']:.2f} | {t['viewer_context_wall_seconds']:.2f} | {t['total_before_serialization_wall_seconds']:.2f} |")
    lines += ['', 'Single sequential runs. Build, JSON serialization and viewer publication excluded. Ceres time is contained in window wall time, which is contained in read/fit/diagnostic time; do not add overlapping columns.', '',
              '## Residual costs', '', 'Ceres half-squared weighted residual sums, evaluated once on the assembled trajectory. Different residual scales make total objective costs unsuitable for ranking accuracy.', '',
              '| Family | '+' | '.join(report)+' |', '| --- | '+' | '.join(['---:']*len(report))+' |']
    for family in next(iter(report.values()))['residual_costs']:
        values=[item['residual_costs'][family] for item in report.values()]
        lines.append('| '+family+' | '+' | '.join(f'{sum(v) if isinstance(v,list) else v:.4f}' for v in values)+' |')
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    prepare_output()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--publish-only',action='store_true')
    args=parser.parse_args()
    path=recording_path('sample');source_sha=digest(path);native_sha=digest(Path(_native.__file__))
    # Frame range from the verified previous full-sample result, without another Parquet decode.
    previous=json.loads((OUTPUT.parent/'axial_axes/shoulder_axis.json').read_text(encoding='utf-8'))
    if previous['metadata']['recording']['sha256']!=source_sha:raise ValueError('Prepared sample changed; regenerate the axial-axis baseline first')
    end=previous['runs'][0]['methods']['full_body']['frames'][-1]['diagnostics']['Recording frame']
    del previous
    OUTPUT.mkdir(parents=True,exist_ok=True)
    candidates={};report={}
    published_native_sha=None
    for name,label,rest,bounds,twist in CASES:
        cache=OUTPUT/(name+'.json')
        spec=dict(name=name,rest=rest,bounds=list(bounds) if bounds else None,twist=twist,source_sha=source_sha,native_sha=native_sha)
        if args.publish_only or (args.resume and cache.exists()):
            candidate=json.loads(cache.read_text(encoding='utf-8'))
            if args.publish_only:
                saved_sha=candidate['runs'][0]['methods']['full_body']['benchmark_case']['native_sha']
                if published_native_sha is None:published_native_sha=saved_sha
                if saved_sha!=published_native_sha:raise ValueError('Published fits used different native binaries')
                spec['native_sha']=published_native_sha
            if candidate['runs'][0]['methods']['full_body']['benchmark_case']!=spec:raise ValueError('Cached case differs: '+name)
        else:
            def progress(w,n):
                if w['index']%200==0 or w['index']==n-1:print(f"{name}: {w['index']+1}/{n}, {w['seconds']:.3f}s",flush=True)
            def solve(**kwargs):
                return fit_windows(kwargs,active_frames=3,max_iterations=MAX_ITERATIONS,function_tolerance=FUNCTION_TOLERANCE,progress=progress)
            print('START '+name,flush=True);started=perf_counter()
            candidate=recording_body_catalog(path,0,end,free_axial_lengths=True,free_length_rest_prior=True,
                length_prior_fraction=rest,chest_line_prior=True,shoulder_profile='lower_closer_sc',relaxed_shoulders=True,
                proportional_spine_lengths=True,spine_proportion_ratios=RATIOS,shared_spine_length=True,
                shoulder_axis_preference=True,shared_total_bound_fractions=bounds,spine_twist_scale=twist,solve_sequence=solve)
            fit_wall=perf_counter()-started;context_start=perf_counter()
            add_context(candidate);add_review_metrics(candidate);add_axis_geometry(candidate);add_twist_diagnostics(candidate)
            method=candidate['runs'][0]['methods']['full_body']
            method['benchmark_case']=spec
            method['benchmark_timing']=dict(ceres_seconds=method['frames'][0]['seconds'],window_processing_wall_seconds=method['processing']['wall_seconds'],
                read_fit_diagnostics_wall_seconds=fit_wall,viewer_context_wall_seconds=perf_counter()-context_start,
                total_before_serialization_wall_seconds=perf_counter()-started,
                scope='Sequential runs on one machine; build, JSON serialization and viewer publication excluded; no repeated-run timing confidence interval.')
        method=candidate['runs'][0]['methods']['full_body'];method['comparison_label']=label
        report[name]=metrics(candidate)
        cache.write_text(json.dumps(candidate,allow_nan=False),encoding='utf-8')
        candidates[name]=candidate
        (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('DONE '+name+' '+json.dumps(report[name]['summary']),flush=True)
    publish(candidates,comparison='spine_stability')
    write_report(report)


if __name__=='__main__':main()
