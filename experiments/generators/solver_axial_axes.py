"""Viewer diagnostics; numerical preparation lives in the fitting package."""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.core.skeleton.fitting.axis_priors import (AXIS_PRIOR_SCALE, AXIS_MINIMUM_LENGTH_MM, PAIRS, observed_pair, shoulder_axis_prior)

AXIS_DISPLAY_HALF_LENGTH_MM = 90.

def add_axis_geometry(candidate):
    definitions = candidate['bodies']
    for method in candidate['runs'][0]['methods'].values():
        sources = method['settings']['direct_mapping_sources']
        for frame in method['frames']:
            axes = []; observed = {}; fitted = {}
            def line(name, a, b, color):
                axes.append(dict(label=name, start=np.asarray(a).tolist(), end=np.asarray(b).tolist(), color=color))
            def axis(name, a, b, color):
                center=(a+b)/2; direction=(b-a)/np.linalg.norm(b-a)
                line(name, center-AXIS_DISPLAY_HALF_LENGTH_MM*direction, center+AXIS_DISPLAY_HALF_LENGTH_MM*direction, color)
                return center, direction
            for name, pair_names in PAIRS.items():
                pair=observed_pair(frame['recording_context']['keypoints'],sources,pair_names)
                if pair is not None: observed[name]=axis('Keypoint-derived '+name+' axis', *pair, '#f2d16b')
            for segment, pair_names, color in (
                ('pelvis',PAIRS['hip'],'#7adea4'),
                ('thoracic',('left_sternoclavicular','right_sternoclavicular'),'#fc945d'),
                ('skull',('left_ear','right_ear'),'#c697ff')):
                index=next(i for i,b in enumerate(definitions) if b['id']==segment)
                body=frame['bodies'][index]; names=definitions[index]['landmark_names']
                a,b=[np.asarray(body['fitted'][names.index(n)]) for n in pair_names]
                fitted[segment]=axis('Fitted '+segment+' lateral axis',a,b,color)
                if segment=='thoracic':
                    rotation=Rotation.from_quat(body['quaternion'],scalar_first=True)
                    if 'shoulder' in observed:
                        local=rotation.inv().apply(observed['shoulder'][1])
                        frame['diagnostics']['Shoulder versus SC axial angle (degrees)']=float(np.degrees(np.arctan2(local[1],local[0])))
                if segment=='skull':
                    head=np.asarray(body['fitted'][names.index('head_center')]); nose=np.asarray(body['fitted'][names.index('nose')])
                    line('Fitted rigid skull: head center to nose',head,nose,color)
            if 'hip' in observed and 'shoulder' in observed:
                line('Keypoint-derived hip center to shoulder center',observed['hip'][0],observed['shoulder'][0],'#f2d16b')
            if 'shoulder' in observed:
                line('Shoulder center to fitted rigid skull center',observed['shoulder'][0],head,'#c697ff')
            for first,second in [('pelvis','thoracic'),('thoracic','skull')]:
                angle=np.degrees(np.arccos(np.clip(np.dot(fitted[first][1],fitted[second][1]),-1,1)))
                frame['diagnostics'][first+' to '+second+' lateral-axis angle (degrees)']=float(angle)
            frame['axis_geometry']=axes
        values=[f['diagnostics']['Shoulder versus SC axial angle (degrees)'] for f in method['frames']
                if 'Shoulder versus SC axial angle (degrees)' in f['diagnostics']]
        if values:
            method['summary']['SC versus shoulder axial angle RMS (degrees)']=float(np.sqrt(np.mean(np.square(values))))
            method['summary']['SC versus shoulder absolute axial angle p95 (degrees)']=float(np.percentile(np.abs(values),95))
    return candidate
