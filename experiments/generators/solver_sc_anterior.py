"""Viewer diagnostics; numerical preparation lives in the fitting package."""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import numpy as np
from skellyforge import _native
from skellyforge.core.skeleton.fitting.sc_prior import (SC_ANTERIOR_SCALE_MM, GEOMETRY_TOLERANCE_MM, audit_cervical_attachment, sc_prior)

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
