"""Fit the complete prepared recording; retain a controlled equality comparison."""
import copy
import argparse
import json
import re
from pathlib import Path
from time import perf_counter
import numpy as np
from scripts.recording_data import read_recording,recording_path
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_spine_equality import length_diagnostics
from scripts.generate_solver_viewer import render_experiments
from scripts.generate_recording_comparison import FOLDER,main as generate_comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-control',action='store_true',help='Also fit the full recording without the equality residual')
    args=parser.parse_args()
    path=recording_path();records,_,_=read_recording(path)
    start,end=records[0]['number'],records[-1]['number']
    outputs=FOLDER.parent/'build'/'full_recording';outputs.mkdir(parents=True,exist_ok=True)
    candidates={};wall_times={}
    methods=[('equal_spine_lengths',True)]+([('lower_sc_relaxed',False)] if args.with_control else [])
    for method_id,equality in methods:
        print(f'FULL RECORDING: {method_id}, frames {start}-{end}, {len(records)} frames',flush=True)
        started=perf_counter()
        candidate=recording_body_catalog(path,start,end,free_axial_lengths=True,chest_line_prior=True,
            shoulder_profile='lower_sc',relaxed_shoulders=True,equal_spine_lengths=equality)
        wall_times[method_id]=perf_counter()-started
        add_context(candidate)
        method=candidate['runs'][0]['methods']['full_body'];length_diagnostics(method)
        method['body_definitions']=candidate['bodies'];method['metadata']=candidate['metadata']
        method['summary']['Read and fit wall time (s)']=wall_times[method_id]
        costs=method['settings']['costs_by_family']
        total=sum(sum(v) if isinstance(v,list) else v for v in costs.values())
        if abs(total-method['frames'][0]['costs'][-1])>1e-6:
            raise ValueError('Residual-family costs disagree with Ceres total')
        (outputs/(method_id+'.json')).write_text(json.dumps(candidate,allow_nan=False),encoding='utf-8')
        candidates[method_id]=candidate
        print('SAVED',method_id,'Ceres seconds',method['frames'][0]['seconds'],'read/fit seconds',wall_times[method_id],flush=True)
    publish(candidates,outputs)


def publish(candidates,outputs):
    current=candidates['equal_spine_lengths']
    a=current['runs'][0]['methods']['full_body']
    if 'lower_sc_relaxed' in candidates:
        check_control(current,candidates['lower_sc_relaxed'])
    experiment=copy.deepcopy(current)
    experiment.update(id='recording_full_body',label='18 - Full recording / equal spine length preference',default_method='equal_spine_lengths',
        description='Full recording: lower-SC reference offsets, relaxed shoulder connections, free spine lengths with a soft equal-length preference. An optional no-equality control is shown only if separately fitted.')
    labels={'lower_sc_relaxed':'Lower SC / relaxed shoulders','equal_spine_lengths':'Lower SC / relaxed shoulders + equal spine lengths'}
    experiment['methods']=[dict(id=k,label=labels[k]) for k in candidates]
    experiment['runs'][0]['methods']={k:v['runs'][0]['methods']['full_body'] for k,v in candidates.items()}
    page=FOLDER/'solver_viewer.html'
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    render_experiments(bank=[e for e in bank if e['id']!=experiment['id']]+[experiment])
    generate_comparison()
    report=[]
    for name,method in experiment['runs'][0]['methods'].items():
        lengths=np.asarray([f['lengths'] for f in method['frames']])
        report.append(dict(method=name,summary=method['summary'],ceres_seconds=method['frames'][0]['seconds'],
            ceres_report=method['frames'][0]['report'],length_min_mm=lengths.min(axis=0).tolist(),length_max_mm=lengths.max(axis=0).tolist(),
            unsupported_frames=[f['diagnostics']['Recording frame'] for f in method['frames'] if f['diagnostics']['Mapped keypoint targets']==0],
            zero_length_frames=[f['diagnostics']['Recording frame'] for f in method['frames'] if 'Spine fit warning' in f['diagnostics']]))
    (outputs/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def check_control(current,reference):
    a=current['runs'][0]['methods']['full_body'];b=reference['runs'][0]['methods']['full_body']
    if current['bodies']!=reference['bodies'] or current['runs'][0]['times']!=reference['runs'][0]['times'] or current['metadata']['recording']['sha256']!=reference['metadata']['recording']['sha256']:
        raise ValueError('Full recording comparison inputs differ')
    for key,value in a['settings'].items():
        if key not in ('costs_by_family','length_equality_prior') and value!=b['settings'][key]:
            raise ValueError('Comparison changed another setting: '+key)
    for af,bf in zip(a['frames'],b['frames'],strict=True):
        for aa,bb in zip(af['bodies'],bf['bodies'],strict=True):
            if any(aa[k]!=bb[k] for k in ('observed','initial_quaternion','initial_translation')):
                raise ValueError('Comparison targets or initialization differ')


if __name__=='__main__':
    main()
