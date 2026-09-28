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


