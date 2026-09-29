"""Synthetic tree inputs; no experiment execution or output generation."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import axial_points


def tree_inputs(noise=1., deforming=False):
    local=np.array(list(itertools.product([-15.,15.],[-15.,15.],[-40.,40.])))
    parents=[0,1,2,2]
    attachments=np.array([[0.,0.,40.],[0.,0.,40.],[0.,-35.,40.],[0.,35.,40.]])
    child=np.tile([0.,0.,-40.],(4,1))
    times=np.linspace(0,2,21)
    rng=np.random.default_rng(17)
    records=[]
    for time in times:
        phase=np.pi*time
        angles=[.1*np.sin(phase),.15*np.sin(phase),.2*np.sin(phase),.7+.25*np.sin(phase),-.7-.25*np.cos(phase)]
        rotations=[Rotation.from_quat([np.cos(a/2),np.sin(a/2),0.,0.],scalar_first=True) for a in angles]
        references=[0.,80.,80.,0.,0.]
        lengths=[0.,80.-24*np.sin(phase/2)**2,80.-16*np.sin(phase/2)**2,0.,0.] if deforming else references
        translations=[np.array([20*np.sin(phase),0.,70.])]
        for b,p in enumerate(parents,1):
            translations.append(translations[p]+rotations[p].apply(axial_points(attachments[b-1],references[p],lengths[p]))-rotations[b].apply(axial_points(child[b-1],references[b],lengths[b])))
        truth=np.array([r.apply(axial_points(local,references[b],lengths[b]))+t for b,(r,t) in enumerate(zip(rotations,translations))])
        records.append(dict(lengths=lengths,rotations=rotations,translations=translations,truth=truth,observed=truth+rng.normal(0,noise,truth.shape)))
    return local,parents,attachments,child,times,records
