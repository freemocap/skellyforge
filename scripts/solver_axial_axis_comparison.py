"""Real sample comparison: shared lengths, with/without thoracic axis residual."""
import argparse
import json
from pathlib import Path
from scripts.recording_data import recording_path, digest
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_rest_length_comparison import add_review_metrics
from scripts.solver_processing_comparison import publish
from scripts.solver_window_sequence import fit_windows
from scripts.solver_axial_axes import add_axis_geometry
from scripts.solver_shared_spine_length import RATIOS, REST_LENGTH_FRACTION, FUNCTION_TOLERANCE, MAX_ITERATIONS, length_metrics

OUTPUT=Path(__file__).resolve().parents[1]/'build/axial_axes'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only',action='store_true');args=parser.parse_args()
    path=recording_path('sample')
    baseline=json.loads((OUTPUT.parent/'shared_spine_length/shared.json').read_text(encoding='utf-8'))
    if baseline['metadata']['recording']['sha256']!=digest(path):raise ValueError('Baseline prepared recording changed')
    def progress(w,n):
        if w['index']%100==0 or w['index']==n-1:print(f"Window {w['index']+1}/{n}: {w['seconds']:.3f}s",flush=True)
    def solve(**kwargs):
        return fit_windows(kwargs,active_frames=3,max_iterations=MAX_ITERATIONS,function_tolerance=FUNCTION_TOLERANCE,progress=progress)
    if args.publish_only:
        candidate=json.loads((OUTPUT/'shoulder_axis.json').read_text(encoding='utf-8'))
    else:
        end=baseline['runs'][0]['methods']['full_body']['frames'][-1]['diagnostics']['Recording frame']
        candidate=recording_body_catalog(path,0,end,free_axial_lengths=True,free_length_rest_prior=True,
            length_prior_fraction=REST_LENGTH_FRACTION,chest_line_prior=True,shoulder_profile='lower_closer_sc',
            relaxed_shoulders=True,proportional_spine_lengths=True,spine_proportion_ratios=RATIOS,
            shared_spine_length=True,shoulder_axis_preference=True,solve_sequence=solve)
        add_context(candidate);add_review_metrics(candidate)
    for value in (baseline,candidate):add_axis_geometry(value)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'shoulder_axis.json').write_text(json.dumps(candidate,allow_nan=False),encoding='utf-8')
    publish({'shared':baseline,'shoulder_axis':candidate},comparison='axial_axes')
    report={name:dict(summary=value['runs'][0]['methods']['full_body']['summary'],length_variation=length_metrics(value))
            for name,value in [('shared',baseline),('shoulder_axis',candidate)]}
    (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
