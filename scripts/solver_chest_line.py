"""Viewer diagnostics; numerical preparation lives in the fitting package."""
import numpy as np
from skellyforge.core.skeleton.fitting.centerline import (CHEST_LINE_DISTANCE_SCALE_MM, CHEST_LINE_ANTERIOR_SCALE_MM, LINE_MINIMUM_EXTENT_MM, LINE_MINIMUM_BASIS_SINE, LINE_SOURCE_LANDMARKS, mapped_centerline)

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
