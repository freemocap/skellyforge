"""Keep connected axial landmarks near their saved post-hoc landmark positions.

These are correlated geometric preferences, not additional measured keypoints.
The acceptance distance is a diagnostic gate, not a Ceres parameter bound.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from skellyforge import _native
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from scripts.recording_data import recording_path, read_recording
from scripts.solver_recording_body import body_model, recording_body_catalog
from scripts.solver_recording_context import add_context
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows
from scripts.solver_shared_spine_length import RATIOS, FUNCTION_TOLERANCE, MAX_ITERATIONS
from scripts.solver_spine_stability import BASE_REST_SCALE, TWIST_SCALE
from scripts.generate_recording_comparison import comparison_data, write_comparison

LANDMARKS = ('pelvis_origin', 'chest_center', 'neck_center', 'craniocervical_junction')
POSITION_SCALE_MM = 5.0
KEYPOINT_HUBER_SCALE_MM = 30.0
ACCEPTANCE_DISTANCE_MM = 50.0  # experimental maximum; not an anatomical claim
OUTPUT = Path(__file__).resolve().parents[1] / 'build/spine_positions'


def position_priors(model, records, scale=POSITION_SCALE_MM):
    priors=[]
    for name in LANDMARKS:
        b=next(i for i,keys in enumerate(model['display_names']) if name in keys)
        slot=model['display_names'][b].index(name)
        prior=_native.LandmarkPositionPrior()
        prior.segment=b;prior.local_point=model['display'][b][slot].tolist()
        prior.scale=scale
        # Do not silently substitute another landmark or a fabricated measurement.
        prior.targets=[r['points'][name].tolist() for r in records]
        priors.append(prior)
    return priors


def audit(candidate, records):
    method=candidate['runs'][0]['methods']['full_body'];report={}
    for name in LANDMARKS:
        b=next(i for i,body in enumerate(candidate['bodies']) if name in body['landmark_names'])
        slot=candidate['bodies'][b]['landmark_names'].index(name)
        distances=[]
        for frame,record in zip(method['frames'],records,strict=True):
            distance=float(np.linalg.norm(np.asarray(frame['bodies'][b]['fitted'][slot])-record['points'][name]))
            if not np.isfinite(distance):
                raise ValueError(f'Nonfinite axial position: {name}, frame {record["number"]}')
            frame['diagnostics'][name+' saved-landmark distance (mm)']=distance
            distances.append(distance)
        report[name]=dict(rms_mm=float(np.sqrt(np.mean(np.square(distances)))),
            maximum_mm=max(distances),worst_frame=records[int(np.argmax(distances))]['number'],
            final_frame_mm=distances[-1],outside_limit_frames=[r['number'] for r,d in zip(records,distances) if d>ACCEPTANCE_DISTANCE_MM])
    passed=all(not v['outside_limit_frames'] for v in report.values())
    method['summary']['Axial position acceptance passed']=passed
    method['summary']['Axial position acceptance distance (mm)']=ACCEPTANCE_DISTANCE_MM
    method['summary']['Maximum axial landmark deviation (mm)']=max(v['maximum_mm'] for v in report.values())
    method['position_acceptance']=dict(passed=passed,limit_mm=ACCEPTANCE_DISTANCE_MM,landmarks=report)
    return method['position_acceptance']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',choices=('test','sample'),default='sample')
    parser.add_argument('--inspect-windows',type=int,nargs='*',default=[])
    args=parser.parse_args()
    output=OUTPUT if args.recording=='sample' else OUTPUT/'test'
    page='recording_spine_positions.html' if args.recording=='sample' else 'recording_test_spine_positions.html'
    path=recording_path(args.recording);records,scale,meta=read_recording(path,include_model=True)
    skeleton=SkeletonSnapshot.from_dict(meta['skeleton']).restore()
    model=body_model(skeleton,meta['model'],scale.segment_scales,'lower_closer_sc',flexible_cervical=True)
    priors=position_priors(model,records)
    inspected={}
    def solve(**kwargs):
        kwargs['landmark_position_priors']=priors
        kwargs['landmark_huber_scale_mm']=KEYPOINT_HUBER_SCALE_MM
        result=fit_windows(kwargs,active_frames=3,max_iterations=MAX_ITERATIONS,
            function_tolerance=FUNCTION_TOLERANCE,
            inspect_windows=args.inspect_windows,
            progress=lambda w,n: print('Window',w['index']+1,'/',n,flush=True) if w['index']%200==0 else None)
        inspected.update(result.inspected_windows)
        return result
    started=perf_counter()
    candidate=recording_body_catalog(path,records[0]['number'],records[-1]['number'],
        free_axial_lengths=True,free_length_rest_prior=True,length_prior_fraction=BASE_REST_SCALE,
        chest_line_prior=True,shoulder_profile='lower_closer_sc',relaxed_shoulders=True,
        proportional_spine_lengths=True,spine_proportion_ratios=RATIOS,
        shoulder_axis_preference=True,spine_twist_scale=TWIST_SCALE,solve_sequence=solve)
    method=candidate['runs'][0]['methods']['full_body']
    method['settings']['landmark_position_priors']=dict(landmarks=LANDMARKS,scale_mm=POSITION_SCALE_MM,
        source='Saved post-hoc landmark positions; correlated geometric preferences, not independent measurements')
    method['settings']['landmark_huber_scale_mm']=KEYPOINT_HUBER_SCALE_MM
    method['objective']+=' Explicit XYZ position residuals on saved axial landmark positions; these are soft geometric preferences. All three axial lengths remain independently adjustable with existing rest and ratio preferences. Keypoint residuals use a 30 mm Huber transition.'
    method['summary']['Read and fit wall seconds']=perf_counter()-started
    result=audit(candidate,records)
    output.mkdir(parents=True,exist_ok=True)
    if inspected:
        from skellyforge.tools.solver_inspector.inspection import inspection_data
        inspection_folder=Path(__file__).parent/'.solver_inspections'/args.recording
        inspection_folder.mkdir(parents=True,exist_ok=True)
        windows=[]
        for index,snapshot in inspected.items():
            snapshot['segment_names']=model['names']
            filename=f'window_{index}.json'
            (inspection_folder/filename).write_text(json.dumps(inspection_data(snapshot),allow_nan=False))
            windows.append(dict(index=index,file=filename,frame_start=snapshot['frame_start'],
                active_start=snapshot['active_start'],frame_end=snapshot['frame_end']))
        (inspection_folder/'manifest.json').write_text(json.dumps(dict(recording=args.recording,
            source=candidate['metadata'],windows=windows),indent=2))
    print('Axial acceptance',json.dumps(result),flush=True)
    (output/'candidate_metrics.json').write_text(json.dumps(result,indent=2))
    add_context(candidate)
    (output/'candidate.json').write_text(json.dumps(candidate,allow_nan=False))
    sources=[];previous=None
    if args.recording=='sample':
        baseline=json.loads((OUTPUT.parent/'fixed_neck/fixed_neck.json').read_text())
        if baseline['metadata']['recording']['sha256']!=candidate['metadata']['recording']['sha256']:
            raise ValueError('Baseline uses another recording')
        previous=audit(baseline,records)
        sources.append(('baseline','Previous fixed neck',baseline))
    sources.append(('anchored','Flexible spine + axial positions + robust keypoints',candidate))
    experiment={**candidate,'methods':[],'runs':[{**candidate['runs'][0],'methods':{}}]}
    for name,label,source in sources:
        m=source['runs'][0]['methods']['full_body']
        m['body_definitions']=source['bodies'];m['metadata']=source['metadata']
        experiment['methods'].append(dict(id=name,label=label))
        experiment['runs'][0]['methods'][name]=m
    write_comparison(comparison_data([experiment]),page,
        '<a href="recording_spine_positions.html">Sample recording</a> · <a href="recording_test_spine_positions.html">Test recording</a>')
    (output/'metrics.json').write_text(json.dumps(dict(baseline=previous,candidate=result),indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError('Axial position acceptance failed; comparison saved for diagnosis, not an accepted fit')


if __name__=='__main__':main()
