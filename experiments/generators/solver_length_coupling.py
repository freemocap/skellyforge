"""Separate exact length coupling, total bounds and soft rest-length preference."""
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
from experiments.generators.solver_sc_anterior import add_sc_diagnostics
from experiments.generators.solver_spine_stability import metrics, BASE_REST_SCALE, STRONG_REST_SCALE, TWIST_SCALE
from experiments.generators.solver_shared_spine_length import RATIOS, FUNCTION_TOLERANCE, MAX_ITERATIONS
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows
from experiments.generators.solver_processing_comparison import publish

OUTPUT = REPO_ROOT/'build/length_coupling'
CASES = (
    ('shared_unbounded', 'Exact ratio / no total bounds', True, BASE_REST_SCALE),
    ('independent', 'Independent lengths / soft ratio / rest 50%', False, BASE_REST_SCALE),
    ('independent_stronger_rest', 'Independent lengths / soft ratio / rest 25%', False, STRONG_REST_SCALE),
)
REVIEW_INTERVAL = (930, 991)  # Includes frame 960; stop exclusive.


def summarize(candidate):
    method=candidate['runs'][0]['methods']['full_body']
    b=next(i for i,v in enumerate(candidate['bodies']) if v['id']=='thoracic')
    slot=candidate['bodies'][b]['landmark_names'].index('neck_center')
    rows=[]
    for f in method['frames']:
        if f.get('chest_line') is None:continue
        delta=np.asarray(f['bodies'][b]['fitted'][slot])-f['chest_line']['shoulder']
        distance=float(np.linalg.norm(delta))
        f['diagnostics']['Neck center to shoulder midpoint (mm)']=distance
        rows.append((f['diagnostics']['Recording frame'],distance))
    selected=[d for n,d in rows if REVIEW_INTERVAL[0]<=n<REVIEW_INTERVAL[1]]
    method['summary']['Neck to shoulder midpoint RMS frames 930-990 (mm)']=float(np.sqrt(np.mean(np.square(selected))))
    result=metrics(candidate)
    result['neck_midpoint_distances']=rows
    result['interpretation']='Geometric agreement with mapped shoulder midpoint, not anatomical ground truth; no new midpoint residual.'
    return result


def main():
    prepare_output()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    path=recording_path('sample'); source_sha=digest(path); native_sha=digest(Path(_native.__file__))
    baseline=json.loads((OUTPUT.parent/'sc_anterior/baseline.json').read_text())
    method=baseline['runs'][0]['methods']['full_body']; identity=method['benchmark_case']
    if identity['source_sha']!=source_sha or identity['native_sha']!=native_sha:
        raise ValueError('Regenerate the current baseline first')
    start,end=[method['frames'][i]['diagnostics']['Recording frame'] for i in (0,-1)]
    method['comparison_label']='Current exact ratio / bounded total / rest 50%'
    candidates={'current':baseline};report={'current':summarize(baseline)}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,label,shared,rest in CASES:
        spec=dict(source_sha=source_sha,native_sha=native_sha,shared=shared,rest_fraction=rest)
        cache=OUTPUT/(name+'.json')
        if args.resume and cache.exists():
            candidate=json.loads(cache.read_text())
            if candidate['runs'][0]['methods']['full_body']['benchmark_case']!=spec:
                raise ValueError('Cached fit configuration differs')
        else:
            def progress(w,n):
                if w['index']%200==0 or w['index']==n-1:print(name,w['index']+1,'/',n,flush=True)
            def solve(**kwargs):
                return fit_windows(kwargs,active_frames=3,max_iterations=MAX_ITERATIONS,
                    function_tolerance=FUNCTION_TOLERANCE,progress=progress)
            started=perf_counter()
            candidate=recording_body_catalog(path,start,end,free_axial_lengths=True,free_length_rest_prior=True,
                length_prior_fraction=rest,chest_line_prior=True,shoulder_profile='lower_closer_sc',
                relaxed_shoulders=True,proportional_spine_lengths=True,spine_proportion_ratios=RATIOS,
                shared_spine_length=shared,shoulder_axis_preference=True,spine_twist_scale=TWIST_SCALE,
                solve_sequence=solve)
            fit_wall=perf_counter()-started; context=perf_counter()
            add_context(candidate);add_review_metrics(candidate);add_axis_geometry(candidate)
            add_twist_diagnostics(candidate);add_sc_diagnostics(candidate)
            method=candidate['runs'][0]['methods']['full_body']
            method['benchmark_case']=spec
            method['benchmark_timing']=dict(ceres_seconds=method['frames'][0]['seconds'],
                window_processing_wall_seconds=method['processing']['wall_seconds'],
                read_fit_diagnostics_wall_seconds=fit_wall,viewer_context_wall_seconds=perf_counter()-context,
                total_before_serialization_wall_seconds=perf_counter()-started)
        method=candidate['runs'][0]['methods']['full_body'];method['comparison_label']=label
        report[name]=summarize(candidate);candidates[name]=candidate
        cache.write_text(json.dumps(candidate,allow_nan=False))
        (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('DONE',name,json.dumps(report[name]['summary']),flush=True)
    publish(candidates,comparison='length_coupling')


if __name__=='__main__':main()
