"""Viewer diagnostics; numerical preparation lives in the fitting package."""
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.core.skeleton.fitting.spine_priors import (SPINE_LINKS, TWIST_AXIS, TWIST_DECOMPOSITION_MINIMUM_NORM, twist_priors)

def add_twist_diagnostics(candidate):
    names=[b['id'] for b in candidate['bodies']]
    method=candidate['runs'][0]['methods']['full_body']
    values=[];undefined=0
    for frame in method['frames']:
        for parent,child in SPINE_LINKS:
            a,b=names.index(parent),names.index(child)
            qa,qb=[Rotation.from_quat(frame['bodies'][i]['quaternion'],scalar_first=True) for i in (a,b)]
            reference=Rotation.from_quat(method['settings']['rest_relative_quaternions'][b-1],scalar_first=True)
            q=(reference.inv()*qa.inv()*qb).as_quat(scalar_first=True)
            if q[0]<0:q=-q
            norm=np.hypot(q[0],q[3])
            angle=None if norm<TWIST_DECOMPOSITION_MINIMUM_NORM else float(np.degrees(2*np.arctan2(q[3],q[0])))
            if angle is None:undefined+=1
            else:values.append(angle)
            frame['diagnostics'][f'Relative axial twist {parent} to {child} (degrees)']=angle
    method['summary']['Relative spine twist RMS (degrees)']=float(np.sqrt(np.mean(np.square(values)))) if values else None
    method['summary']['Relative spine absolute twist p95 (degrees)']=float(np.percentile(np.abs(values),95)) if values else None
    method['summary']['Undefined twist decompositions']=undefined
