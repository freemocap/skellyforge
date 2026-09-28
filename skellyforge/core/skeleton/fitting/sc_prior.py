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


