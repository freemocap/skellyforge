"""Compare exact vs penalized shoulder attachments, with fixed person scale."""
import copy
import json
import re
from pathlib import Path
from scripts.generate_solver_viewer import render_experiments
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context


def main():
    page=Path(__file__).with_name('solver_viewer.html')
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    previous=next(e for e in bank if e['id']=='recording_shoulders')
    experiment=copy.deepcopy(previous)
    experiment.update(id='recording_shoulder_linkages',default_method='lower_sc_relaxed',label='16 - Real recording / relaxed shoulder connections',
        description='Exact versus penalized clavicle-to-upper-arm attachment coincidence. Same fixed person scale and rigid clavicle/arm geometry. Yellow lines show fitted separation; this is an effective linkage experiment, not separately measured joint centers.')
    experiment['methods']=[];experiment['runs'][0]['methods']={}
    path=Path(previous['metadata']['recording']['path'])
    for profile in ('lower_sc','lower_closer_sc'):
        baseline=previous['runs'][0]['methods'][profile]
        first,last=baseline['frames'][0],baseline['frames'][-1]
        candidate=recording_body_catalog(path,first['diagnostics']['Recording frame'],last['diagnostics']['Recording frame'],
            free_axial_lengths=True,chest_line_prior=True,shoulder_profile=profile,relaxed_shoulders=True)
        add_context(candidate)
        method=candidate['runs'][0]['methods']['full_body']
        if (candidate['metadata']['recording']['sha256']!=previous['metadata']['recording']['sha256']
                or candidate['runs'][0]['times']!=previous['runs'][0]['times']
                or candidate['bodies']!=baseline['body_definitions']):
            raise ValueError('Shoulder linkage comparison changed recording, timestamps or reference geometry')
        for k in ('shoulder_geometry','axial_reference_lengths','direct_mapping_sources','rest_relative_quaternions',
                  'chest_line_prior','free_axial_lengths','position_scale_mm','linear_motion_scale','angular_motion_scale','rest_pose_scale_radians'):
            if json.loads(json.dumps(method['settings'][k]))!=baseline['settings'][k]:
                raise ValueError('Shoulder linkage comparison changed another setting: '+k)
        for a,b in zip(baseline['frames'],method['frames'],strict=True):
            for aa,bb in zip(a['bodies'],b['bodies'],strict=True):
                if any(aa[k]!=bb[k] for k in ('observed','initial_quaternion','initial_translation')):
                    raise ValueError('Shoulder linkage comparison changed targets or initialization')
        method['body_definitions']=candidate['bodies'];method['metadata']=candidate['metadata']
        label='Lower SC' if profile=='lower_sc' else 'Lower + closer SC'
        for key,value,suffix in ((profile,baseline,'exact shoulders'),(profile+'_relaxed',method,'relaxed shoulders')):
            experiment['methods'].append(dict(id=key,label=label+' / '+suffix))
            experiment['runs'][0]['methods'][key]=value
    render_experiments(bank=[e for e in bank if e['id']!=experiment['id']]+[experiment])


if __name__=='__main__':
    main()
