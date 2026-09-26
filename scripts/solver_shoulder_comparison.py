"""Compare fixed SC reference offsets with the accepted chest-line fit."""
import copy
import json
import re
from pathlib import Path
from scripts.generate_solver_viewer import render_experiments
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_shoulder_offsets import SHOULDER_PROFILES, shoulder_diagnostics


def validate_comparison(baseline, candidate):
    a=baseline['runs'][0];b=candidate['runs'][0]
    if a['times']!=b['times'] or baseline['metadata']['recording']['sha256']!=candidate['metadata']['recording']['sha256']:
        raise ValueError('Shoulder comparison requires identical recording and timestamps')
    schema=lambda e:[(d['id'],d['landmark_names'],d['region'],d['edges']) for d in e['bodies']]
    if schema(baseline)!=schema(candidate):
        raise ValueError('Shoulder comparison cannot change segment/landmark identities')
    old=a['methods']['chest_line'];new=b['methods']['full_body']
    if old['problem']!=new['problem']:
        raise ValueError('Shoulder comparison cannot change the Ceres problem structure')
    for k in ('chest_line_prior','free_axial_lengths','axial_length_bound_fractions',
              'axial_reference_lengths','length_prior_fraction','lengthening_prior_fraction',
              'length_acceleration_scale','position_scale_mm','linear_motion_scale',
              'angular_motion_scale','rest_pose_scale_radians','rest_relative_quaternions','direct_mapping_sources'):
        if old['settings'][k]!=json.loads(json.dumps(new['settings'][k])):
            raise ValueError('Shoulder comparison changed another setting: '+k)
    for af,bf in zip(old['frames'],new['frames'],strict=True):
        if af['bodies'][0]['initial_translation']!=bf['bodies'][0]['initial_translation']:
            raise ValueError('Root initialization changed')
        for ab,bb in zip(af['bodies'],bf['bodies'],strict=True):
            if any(ab[k]!=bb[k] for k in ('observed','initial_quaternion')):
                raise ValueError('Keypoint targets or quaternion initialization changed')
    # Initial child translations legitimately change: the attachment equations
    # derive them from the changed fixed geometry, before Ceres runs.


def main():
    page=Path(__file__).with_name('solver_viewer.html')
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    baseline=next(e for e in bank if e['id']=='recording_body')
    source=baseline['runs'][0]['methods']['chest_line']
    experiment=copy.deepcopy(baseline)
    experiment.update(id='recording_shoulders',label='15 - Real recording / shoulder attachment offsets',
        description='Same person scale, rigid clavicle lengths, free spine and chest-line residuals. Only fixed SC/notch reference geometry differs. Offsets are experimental, not anatomical defaults.')
    method=copy.deepcopy(source);method['body_definitions']=copy.deepcopy(baseline['bodies'])
    shoulder_diagnostics(method['frames'],method['body_definitions'])
    experiment['methods']=[dict(id='current_sc',label='Current SC attachments / reference')]
    experiment['runs'][0]['methods']={'current_sc':method}
    frames=source['frames'];start=frames[0]['diagnostics']['Recording frame'];end=frames[-1]['diagnostics']['Recording frame']
    path=Path(baseline['metadata']['recording']['path'])
    for profile,spec in SHOULDER_PROFILES.items():
        candidate=recording_body_catalog(path,start,end,free_axial_lengths=True,chest_line_prior=True,shoulder_profile=profile)
        validate_comparison(baseline,candidate)
        add_context(candidate)
        method=candidate['runs'][0]['methods']['full_body']
        method['body_definitions']=candidate['bodies'];method['metadata']=candidate['metadata']
        experiment['runs'][0]['methods'][profile]=method
        experiment['methods'].append(dict(id=profile,label=spec['label']))
    render_experiments(bank=[e for e in bank if e['id']!='recording_shoulders']+[experiment])


if __name__=='__main__':
    main()
