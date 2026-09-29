"""Known synthetic landmark inputs through the accepted human Ceres fit.

This ideal full-landmark fixture is not a simulated COCO detector. Every input
has an explicit one-to-one synthetic keypoint mapping. It tests fitting and
presentation without claiming realistic sparse observation coverage.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from skellyforge.core.skeleton.fitting import fit_human
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.tools.solver_inspector.inspection import inspection_data
from .synthetic import _build_data


def source_hashes():
    from skellyforge import _native
    from skellyforge.core.skeleton import fitting
    paths = [Path(__file__), Path(__file__).with_name('synthetic.py'), Path(_native.__file__)]
    paths.extend(Path(fitting.__file__).parent.glob('*.py'))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def apply_fit(data, captured, fit):
    """Populate the synthetic overlay from a fit, without writing or solving."""
    model, sequence = fit.model, fit.sequence
    for frame, translations, quaternions, lengths in zip(data['frames'], sequence.translations, sequence.quaternions, sequence.lengths, strict=True):
        displacements = []
        for meta, segment in zip(data['segments_meta'], frame['segments'], strict=True):
            name = meta['name']; b = model['names'].index(name)
            endpoint = captured['skeleton'].segments[name].frame_definition.primary_point_name
            local = model['display'][b][model['display_names'][b].index(endpoint)].copy()
            if model['references'][b] > 0:
                local[2] *= lengths[b] / model['references'][b]
            origin = np.asarray(translations[b])
            q = RotationQuaternion.from_array(array=np.asarray(quaternions[b]))
            segment['connected_basis'] = q.to_rotation_matrix().T.tolist()
            segment['connected_origin'] = origin.tolist()
            segment['connected_end'] = (origin + q.rotate_vector(vector=local)).tolist()
            segment['displacement_mm'] = float(np.linalg.norm(origin-np.asarray(segment['origin'])))
            displacements.append(segment['displacement_mm'])
        # Retain existing diagnostics only when they describe the displayed result.
        frame['fit_displacements'] = displacements
    panel = next(p for p in data['timeseries']['panels'] if p['title'].startswith('Connected vs'))
    panel['title'] = 'Ceres vs hydrated segment origins (difference, not ground-truth error)'
    panel['series'][0]['values'] = [float(np.sqrt(np.mean(np.square(f['fit_displacements'])))) for f in data['frames']]
    panel['series'][1]['values'] = [max(f['fit_displacements']) for f in data['frames']]
    data['fit_source_hashes'] = source_hashes()
    data['fit_report'] = sequence.report
    data['fit_processing'] = sequence.processing
    return data


def prepare(output):
    output = Path(output)
    captured = {}
    data = _build_data(root_motion=True, shoulders=True, elbows=True, head=True, torso=True, legs=True, wrists=True, fingers=True, feet=True, capture=captured)
    fit = fit_human(**captured, inspect_windows=(0,))
    apply_fit(data, captured, fit)
    output.mkdir(parents=True, exist_ok=True)
    (output/'synthetic.json').write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
    snapshot = fit.sequence.inspected_windows[0]
    snapshot['segment_names'] = fit.model['names']
    (output/'synthetic_window_0.json').write_text(json.dumps(inspection_data(snapshot), allow_nan=False), encoding='utf-8')
    return data
