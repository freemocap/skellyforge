"""Viewer diagnostics; numerical preparation lives in the fitting package."""
import numpy as np
from skellyforge.core.skeleton.fitting.settings import BOUND_CONTACT_TOLERANCE_MM
from skellyforge.core.skeleton.fitting.reference_geometry import (SC_LOWERING_REFERENCE_FRACTION, SC_REDUCED_FORWARD_FACTOR, SHOULDER_PROFILES, SC_LANDMARKS, reference_positions)

def shoulder_diagnostics(frames, bodies):
    indices = {body['id']: i for i, body in enumerate(bodies)}
    for frame in frames:
        if any(length <= BOUND_CONTACT_TOLERANCE_MM for length in frame['lengths']):
            frame['diagnostics']['Spine fit warning'] = 'A free spine length reached zero; this is not a usable anatomical pose.'
        line = frame.get('chest_line')
        if line is not None:
            up = np.asarray(line['up'])
            shoulder_center = np.asarray(line['shoulder'])
            thoracic = indices['thoracic']
            slot = bodies[thoracic]['landmark_names'].index('neck_center')
            neck = np.asarray(frame['bodies'][thoracic]['fitted'][slot])
            sc = np.mean([frame['bodies'][indices[side+'_clavicle']]['translation']
                          for side in ('left', 'right')], axis=0)
            frame['diagnostics']['Neck center above shoulder midpoint (mm)'] = float((neck-shoulder_center) @ up)
            frame['diagnostics']['SC midpoint above shoulder midpoint (mm)'] = float((sc-shoulder_center) @ up)
            frame['diagnostics']['SC offset from neck along torso up (mm)'] = float((sc-neck) @ up)
    for side in ('left', 'right'):
        index = next(i for i,b in enumerate(bodies) if b['id']==side+'_clavicle')
        slot = bodies[index]['landmark_names'].index(side+'_acromion')
        for frame in frames:
            body=frame['bodies'][index]
            fitted=np.asarray(body['fitted'][slot]);origin=np.asarray(body['translation'])
            frame['diagnostics'][side+' clavicle length (mm)']=float(np.linalg.norm(fitted-origin))
            observed=body['observed'][slot]
            if observed is not None:
                frame['diagnostics'][side+' shoulder target error (mm)']=float(np.linalg.norm(fitted-observed))
                frame['diagnostics'][side+' SC to shoulder keypoint distance (mm)']=float(np.linalg.norm(origin-observed))
