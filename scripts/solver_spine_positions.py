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
from skellyforge.core.skeleton.fitting.human import (
    LANDMARKS, POSITION_SCALE_MM, KEYPOINT_HUBER_SCALE_MM,
    human_fit_options, fit_prepared_human, position_priors,
)
from scripts.generate_recording_comparison import comparison_data, write_comparison

ACCEPTANCE_DISTANCE_MM = 50.0  # experimental maximum; not an anatomical claim
OUTPUT = Path(__file__).resolve().parents[1] / 'build/spine_positions'


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
    model=body_model(skeleton,meta['model'],scale.segment_scales,human_fit_options()['shoulder_profile'],flexible_cervical=True)
    inspected={}
    def solve(**kwargs):
        result=fit_prepared_human(kwargs,model,records,inspect_windows=args.inspect_windows,
            progress=lambda w,n: print('Window',w['index']+1,'/',n,flush=True) if w['index']%200==0 else None)
        inspected.update(result.inspected_windows)
        return result
    started=perf_counter()
    candidate=recording_body_catalog(path,records[0]['number'],records[-1]['number'],
        **human_fit_options(),solve_sequence=solve)
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
