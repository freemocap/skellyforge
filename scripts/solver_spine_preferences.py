"""Explicit experimental preferences on the existing axial chain."""
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native

SPINE_LINKS = (('pelvis', 'sacrolumbar'), ('sacrolumbar', 'thoracic'),
               ('thoracic', 'cervical_spine'), ('cervical_spine', 'skull'))
TWIST_AXIS = [0., 0., 1.]
TWIST_DECOMPOSITION_MINIMUM_NORM = 1e-10


def twist_priors(model, scale):
    priors=[]
    for parent,child in SPINE_LINKS:
        p=_native.RelativeTwistPrior()
        p.parent=model['names'].index(parent);p.child=model['names'].index(child)
        if model['parents'][p.child-1]!=p.parent:raise ValueError('Spine experiment does not match authored chain')
        p.reference=model['relative'][p.child-1];p.axis=TWIST_AXIS;p.scale=scale
        priors.append(p)
    return priors


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
