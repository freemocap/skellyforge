"""One controlled equal-length comparison against the relaxed-shoulder fit."""
import copy
import json
import re
from pathlib import Path
import numpy as np
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.generate_solver_viewer import render_experiments
from scripts.generate_recording_comparison import comparison_data, FOLDER


def length_diagnostics(method):
    lengths=np.asarray([f['lengths'] for f in method['frames']])
    difference=lengths[:,0]-lengths[:,1]
    method['summary']['Spine length difference RMS (mm)']=float(np.sqrt(np.mean(difference**2)))
    method['summary']['Maximum spine length difference (mm)']=float(np.max(abs(difference)))
    for frame,delta in zip(method['frames'],difference,strict=True):
        frame['diagnostics']['Sacrolumbar minus thoracic (mm)']=float(delta)


def main():
    page=FOLDER/'solver_viewer.html'
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    previous=next(e for e in bank if e['id']=='recording_shoulder_linkages')
    baseline=previous['runs'][0]['methods']['lower_sc_relaxed']
    first,last=baseline['frames'][0],baseline['frames'][-1]
    candidate=recording_body_catalog(Path(previous['metadata']['recording']['path']),
        first['diagnostics']['Recording frame'],last['diagnostics']['Recording frame'],
        free_axial_lengths=True,chest_line_prior=True,shoulder_profile='lower_sc',relaxed_shoulders=True,equal_spine_lengths=True)
    add_context(candidate)
    method=candidate['runs'][0]['methods']['full_body']
    if (candidate['metadata']['recording']['sha256']!=previous['metadata']['recording']['sha256']
            or candidate['runs'][0]['times']!=previous['runs'][0]['times']
            or candidate['bodies']!=baseline['body_definitions'] or method['problem']!=baseline['problem']):
        raise ValueError('Equality comparison changed recording, geometry or parameter structure')
    for key,value in baseline['settings'].items():
        if key not in ('costs_by_family','length_equality_prior') and json.loads(json.dumps(method['settings'][key]))!=value:
            raise ValueError('Equality comparison changed another setting: '+key)
    for a,b in zip(baseline['frames'],method['frames'],strict=True):
        for aa,bb in zip(a['bodies'],b['bodies'],strict=True):
            if any(aa[k]!=bb[k] for k in ('observed','initial_quaternion','initial_translation')):
                raise ValueError('Equality comparison changed targets or initial poses')
    method['body_definitions']=candidate['bodies'];method['metadata']=candidate['metadata']
    experiment=copy.deepcopy(previous)
    experiment.update(id='recording_spine_equality',label='17 - Real recording / equal spine length preference',
        default_method='equal_spine_lengths',description='Same lower-SC relaxed-shoulder fit, with one extra sacrolumbar/thoracic length-difference residual per frame. Total length remains free; no temporal length residual.')
    reference=copy.deepcopy(baseline)
    for value in (reference,method):length_diagnostics(value)
    experiment['methods']=[dict(id='lower_sc_relaxed',label='Lower SC / relaxed shoulders'),dict(id='equal_spine_lengths',label='Lower SC / relaxed shoulders + equal spine lengths')]
    experiment['runs'][0]['methods']={'lower_sc_relaxed':reference,'equal_spine_lengths':method}
    bank=[e for e in bank if e['id']!=experiment['id']]+[experiment]
    # Check overlay compatibility before replacing either viewer artifact.
    comparison_data(bank)
    render_experiments(bank=bank)
    from scripts.generate_recording_comparison import main as generate_comparison
    generate_comparison()
    for label,value in [('Reference',reference),('Equal-length preference',method)]:
        print(label,value['summary'],flush=True)


if __name__=='__main__':
    main()
