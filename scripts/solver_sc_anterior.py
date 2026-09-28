"""One-sided SC preference in the existing keypoint-derived torso frame."""
import numpy as np
from skellyforge import _native

SC_ANTERIOR_SCALE_MM = 20.  # Experimental residual scale, not an anatomical bound.
GEOMETRY_TOLERANCE_MM = 1e-8


def audit_cervical_attachment(model):
    thoracic = model['names'].index('thoracic')
    cervical = model['names'].index('cervical_spine')
    neck = model['display'][thoracic][model['display_names'][thoracic].index('neck_center')]
    if model['parents'][cervical-1] != thoracic or not np.allclose(
            model['attachments'][cervical-1], neck, atol=GEOMETRY_TOLERANCE_MM, rtol=0):
        raise ValueError('Cervical origin must attach exactly to thoracic neck_center')
    # child_attachments are zero and only upper-arm connections are relaxed.


def sc_prior(model, frames, scale):
    prior = _native.LandmarkHalfSpacePrior()
    prior.segment = model['names'].index('thoracic')
    keys = model['display_names'][prior.segment]
    prior.local_point = np.mean([model['display'][prior.segment][keys.index(side+'_sternoclavicular')]
                                for side in ('left', 'right')], axis=0).tolist()
    prior.frames = [None if f is None else [f['shoulder'], f['anterior']] for f in frames]
    prior.scale = scale
    return prior


def add_sc_diagnostics(candidate):
    bodies = candidate['bodies']
    thoracic = next(i for i,b in enumerate(bodies) if b['id']=='thoracic')
    cervical = next(i for i,b in enumerate(bodies) if b['id']=='cervical_spine')
    keys = bodies[thoracic]['landmark_names']
    method = candidate['runs'][0]['methods']['full_body']
    distances=[]; neck_distances=[]; attachment_errors=[]
    for frame in method['frames']:
        fitted=np.asarray(frame['bodies'][thoracic]['fitted'])
        neck=fitted[keys.index('neck_center')]
        attachment_errors.append(float(np.linalg.norm(neck-frame['bodies'][cervical]['translation'])))
        line=frame.get('chest_line')
        if line is None:continue
        sc=np.mean([fitted[keys.index(side+'_sternoclavicular')] for side in ('left','right')],axis=0)
        distance=float((sc-np.asarray(line['shoulder'])) @ line['anterior'])
        neck_distance=float((neck-np.asarray(line['shoulder'])) @ line['anterior'])
        frame['diagnostics']['SC anterior to shoulder midpoint (mm)']=distance
        frame['diagnostics']['Neck anterior to shoulder midpoint (mm)']=neck_distance
        distances.append(distance);neck_distances.append(neck_distance)
    if not distances:raise ValueError('No supported hip/shoulder frames for SC diagnostics')
    violation=np.maximum(-np.asarray(distances),0)
    method['summary'].update({
        'SC posterior violation RMS (mm)':float(np.sqrt(np.mean(violation**2))),
        'SC posterior violation maximum (mm)':float(np.max(violation)),
        'SC posterior frames':int(np.sum(np.asarray(distances)<-GEOMETRY_TOLERANCE_MM)),
        'SC anterior distance minimum (mm)':float(np.min(distances)),
        'SC anterior distance maximum (mm)':float(np.max(distances)),
        'SC anterior distance standard deviation (mm)':float(np.std(distances)),
        'Neck anterior distance standard deviation (mm)':float(np.std(neck_distances)),
        'Cervical attachment error maximum (mm)':float(np.max(attachment_errors)),
    })
