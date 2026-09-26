"""Explicit experimental local-Z deformation, shared by fitting output and viewer data."""
import numpy as np


def axial_references(model):
    references=[0.]*5
    for b,name in [(1,'chest_center'),(2,'neck_center')]:
        references[b]=float(model['display'][b][model['display_names'][b].index(name),2])
        if references[b]<=0:raise ValueError('Spine reference extent must be positive')
    return references


def axial_points(points,reference,length):
    result=np.array(points,dtype=float,copy=True)
    if reference>0:result[...,2]*=length/reference
    return result
