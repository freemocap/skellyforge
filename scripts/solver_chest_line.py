"""Mapped hip/shoulder geometry for an explicit chest-center line preference."""
import numpy as np

# Initial experimental scales, not anatomical limits or estimated noise levels.
CHEST_LINE_DISTANCE_SCALE_MM = 50.
CHEST_LINE_ANTERIOR_SCALE_MM = 20.
LINE_MINIMUM_EXTENT_MM = 1e-6
LINE_MINIMUM_BASIS_SINE = 1e-6
LINE_SOURCE_LANDMARKS = ('left_hip_socket', 'right_hip_socket', 'left_acromion', 'right_acromion')


def mapped_centerline(keypoints, sources):
    names = [sources[k] for k in LINE_SOURCE_LANDMARKS]
    if any(k not in keypoints for k in names):
        return None
    lh, rh, ls, rs = [np.asarray(keypoints[k], dtype=float) for k in names]
    if not np.all(np.isfinite([lh, rh, ls, rs])):
        return None
    origin, shoulder = (lh+rh)/2, (ls+rs)/2
    up, right = shoulder-origin, rh-lh
    if min(np.linalg.norm(up), np.linalg.norm(right)) < LINE_MINIMUM_EXTENT_MM:
        return None
    up /= np.linalg.norm(up)
    anterior = np.cross(up, right/np.linalg.norm(right))
    if np.linalg.norm(anterior) < LINE_MINIMUM_BASIS_SINE:
        return None
    anterior /= np.linalg.norm(anterior)
    lateral = np.cross(anterior, up)
    return dict(origin=origin.tolist(), shoulder=shoulder.tolist(), up=up.tolist(),
                lateral=lateral.tolist(), anterior=anterior.tolist())


def line_diagnostics(frame, geometry, body_index, landmark_index):
    if geometry is None:
        frame['chest_line'] = None
        return
    fitted = np.asarray(frame['bodies'][body_index]['fitted'][landmark_index])
    delta = fitted-np.asarray(geometry['origin'])
    lateral = float(delta @ geometry['lateral'])
    anterior = float(delta @ geometry['anterior'])
    projection = fitted-lateral*np.asarray(geometry['lateral'])-anterior*np.asarray(geometry['anterior'])
    frame['chest_line'] = dict(**geometry, fitted=fitted.tolist(), projection=projection.tolist(),
                              lateral_mm=lateral, anterior_mm=anterior)
    frame['diagnostics'].update({'Chest center lateral from line (mm)': lateral,
                                'Chest center anterior from line (mm)': anterior})
