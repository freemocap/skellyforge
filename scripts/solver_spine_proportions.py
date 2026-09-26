"""Test user-supplied three-spine proportions on the established movement window."""
import json
import re
from time import perf_counter
import numpy as np
from scripts.recording_data import recording_path
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.generate_solver_viewer import render_experiments
from scripts.generate_recording_comparison import FOLDER,main as generate_comparison


def main():
    page=FOLDER/'solver_viewer.html'
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    previous=next(e for e in bank if e['id']=='recording_spine_equality')
    baseline=previous['runs'][0]['methods']['equal_spine_lengths']
    first,last=[baseline['frames'][i]['diagnostics']['Recording frame'] for i in (0,-1)]
    start=perf_counter()
    candidate=recording_body_catalog(recording_path(),first,last,free_axial_lengths=True,
        chest_line_prior=True,shoulder_profile='lower_sc',relaxed_shoulders=True,proportional_spine_lengths=True)
    elapsed=perf_counter()-start
    add_context(candidate)
    method=candidate['runs'][0]['methods']['full_body']
    if candidate['metadata']['recording']['sha256']!=previous['metadata']['recording']['sha256'] or candidate['runs'][0]['times']!=previous['runs'][0]['times'] or candidate['bodies']!=baseline.get('body_definitions',previous['bodies']):
        raise ValueError('Ratio comparison changed source recording, reference geometry or frame times')
    for a,b in zip(method['frames'],baseline['frames'],strict=True):
        for aa,bb in zip(a['bodies'],b['bodies'],strict=True):
            if any(aa[k]!=bb[k] for k in ('observed','initial_quaternion','initial_translation')):
                raise ValueError('Ratio comparison changed targets or initial poses')
    method['body_definitions']=candidate['bodies'];method['metadata']=candidate['metadata']
    method['summary']['Read and fit wall time (s)']=elapsed
    p=method['settings']['length_proportion_prior'];axial=method['problem']['axial_segments']
    lengths=np.array([f['lengths'] for f in method['frames']])[:,[axial.index(b) for b in p['segments']]]
    deviations=lengths-lengths.sum(axis=1)[:,None]*np.array(p['fractions'])
    method['summary']['Proportion deviation RMS (mm)']=float(np.sqrt(np.mean(deviations**2)))
    for f,row in zip(method['frames'],deviations):
        f['diagnostics']['Proportion deviation RMS (mm)']=float(np.sqrt(np.mean(row**2)))
    costs=method['settings']['costs_by_family']
    if abs(sum(sum(v) if isinstance(v,list) else v for v in costs.values())-method['frames'][0]['costs'][-1])>1e-6:
        raise ValueError('Cost family accounting disagrees with Ceres')
    candidate.update(id='recording_spine_proportions',label='19 - Three flexible spine lengths / supplied proportions',default_method='proportional_spine')
    candidate['methods']=[dict(id='proportional_spine',label='Flexible cervical / 6.3 : 20 : 18 spine ratio')]
    candidate['runs'][0]['methods']={'proportional_spine':method}
    output=FOLDER.parent/'build'/'spine_proportions.json'
    output.write_text(json.dumps(candidate,allow_nan=False),encoding='utf-8')
    render_experiments(bank=[e for e in bank if e['id']!=candidate['id']]+[candidate])
    generate_comparison()
    print(method['frames'][0]['report'],json.dumps(method['summary'],indent=2),flush=True)


if __name__=='__main__':main()
