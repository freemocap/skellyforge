"""Named lateral axes from keypoint pairs and existing rigid-segment geometry."""
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native

AXIS_PRIOR_SCALE = .5  # experimental chord scale, roughly 29 degrees near alignment
AXIS_MINIMUM_LENGTH_MM = 1e-6
PAIRS = {'hip': ('left_hip_socket', 'right_hip_socket'),
         'shoulder': ('left_acromion', 'right_acromion')}


def observed_pair(keypoints, sources, pair):
    values = [keypoints.get(sources.get(name)) for name in pair]
    if any(v is None for v in values): return None
    a, b = np.asarray(values, dtype=float)
    if not np.isfinite([a,b]).all() or np.linalg.norm(b-a)<AXIS_MINIMUM_LENGTH_MM: return None
    return a, b


def shoulder_axis_prior(records, model):
    p = _native.SegmentAxisPrior()
    p.segment = model['names'].index('thoracic')
    points = dict(zip(model['display_names'][p.segment], model['display'][p.segment]))
    lateral = points['right_sternoclavicular']-points['left_sternoclavicular']
    lateral /= np.linalg.norm(lateral)
    anterior = np.cross([0.,0.,1.], lateral)
    anterior /= np.linalg.norm(anterior)
    p.local_lateral = lateral.tolist(); p.local_anterior = anterior.tolist()
    p.scale = AXIS_PRIOR_SCALE
    frames = []
    for record in records:
        pair = observed_pair(record['keypoints'], model['sources'], PAIRS['shoulder'])
        frames.append(None if pair is None else ((pair[1]-pair[0])/np.linalg.norm(pair[1]-pair[0])).tolist())
    p.frames = frames
    return p


