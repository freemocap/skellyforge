"""Known synthetic geometry; no fitting or viewer dependencies."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation(axis, angle):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return Rotation.from_quat(
        np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)], scalar_first=True
    )

def axial_points(points,reference,length):
    result=np.array(points,dtype=float,copy=True)
    if reference>0:result[...,2]*=length/reference
    return result
