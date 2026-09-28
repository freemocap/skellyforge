"""Independent spine lengths, with versus without a constant cervical length."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from skellyforge import _native
from scripts.recording_data import recording_path, read_recording, digest
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_rest_length_comparison import add_review_metrics
from scripts.solver_axial_axes import add_axis_geometry
from scripts.solver_spine_preferences import add_twist_diagnostics
from scripts.solver_sc_anterior import add_sc_diagnostics
from scripts.solver_spine_stability import BASE_REST_SCALE, TWIST_SCALE
from scripts.solver_shared_spine_length import RATIOS, FUNCTION_TOLERANCE, MAX_ITERATIONS
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows
from scripts.solver_processing_comparison import publish
from scripts.solver_length_coupling import summarize

OUTPUT=Path(__file__).resolve().parents[1]/'build/fixed_neck'


def main():
    path=recording_path('sample'); records,_,meta=read_recording(path)
    start,end=records[0]['number'],records[-1]['number'];del records
    # The adapter declares the ordering; obtain its checked baseline body order,
    # never infer a native segment index from an anatomical name convention.
    baseline=json.loads((OUTPUT.parent/'length_coupling/independent.json').read_text())
    if baseline['metadata']['recording']['sha256']!=meta['sha256']:
        raise ValueError('Independent comparison belongs to another recording')
    index=next(i for i,b in enumerate(baseline['bodies']) if b['id']=='cervical_spine')
    del baseline
    OUTPUT.mkdir(parents=True,exist_ok=True);candidates={};report={}
    for name,fixed in [('flexible_neck',False),('fixed_neck',True)]:
        def progress(w,n):
            if w['index']%200==0 or w['index']==n-1:print(name,w['index']+1,'/',n,flush=True)
        def solve(**kwargs):
            return fit_windows(kwargs,active_frames=3,max_iterations=MAX_ITERATIONS,
                function_tolerance=FUNCTION_TOLERANCE,progress=progress,
                fixed_length_segments=[index] if fixed else [])
        started=perf_counter()
        candidate=recording_body_catalog(path,start,end,free_axial_lengths=True,
            free_length_rest_prior=True,length_prior_fraction=BASE_REST_SCALE,
            chest_line_prior=True,shoulder_profile='lower_closer_sc',relaxed_shoulders=True,
            proportional_spine_lengths=True,spine_proportion_ratios=RATIOS,
            shoulder_axis_preference=True,spine_twist_scale=TWIST_SCALE,solve_sequence=solve)
        assert candidate['bodies'][index]['id']=='cervical_spine'
        fit_wall=perf_counter()-started;context=perf_counter()
        add_context(candidate);add_review_metrics(candidate);add_axis_geometry(candidate)
        add_twist_diagnostics(candidate);add_sc_diagnostics(candidate)
        method=candidate['runs'][0]['methods']['full_body']
        reference=method['settings']['axial_reference_lengths'][index]
        if fixed:
            np.testing.assert_array_equal([f['diagnostics']['cervical_spine length (mm)'] for f in method['frames']],
                                           [reference]*len(method['frames']))
            method['objective']+=' Cervical length fixed at the recording reference using SetParameterBlockConstant; its quaternion remains free. No neck-midpoint residual.'
        method['comparison_label']=f'Cervical fixed at {reference:.1f} mm' if fixed else 'Independent lengths / flexible cervical'
        method['benchmark_case']=dict(source_sha=meta['sha256'],native_sha=digest(Path(_native.__file__)),fixed_cervical=fixed,reference_cervical_mm=reference)
        method['benchmark_timing']=dict(ceres_seconds=method['frames'][0]['seconds'],window_processing_wall_seconds=method['processing']['wall_seconds'],
            read_fit_diagnostics_wall_seconds=fit_wall,viewer_context_wall_seconds=perf_counter()-context,total_before_serialization_wall_seconds=perf_counter()-started)
        report[name]=summarize(candidate);candidates[name]=candidate
        (OUTPUT/(name+'.json')).write_text(json.dumps(candidate,allow_nan=False))
        (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('DONE',name,json.dumps(report[name]['summary']),flush=True)
    publish(candidates,comparison='fixed_neck')


if __name__=='__main__':main()
