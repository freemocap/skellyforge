"""Compare synthetic hand geometry and angular smoothing; never change defaults."""
import argparse
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from skellyforge.core.skeleton.fitting.body_model import body_model, frame_targets
from skellyforge.core.skeleton.fitting.human import HumanFit, human_fit_options, fit_prepared_human
from skellyforge.core.skeleton.fitting.preparation import prepare_body_fit
from skellyforge.tools.viewer.synthetic import _build_data, HTML_TEMPLATE, _vendored_scripts
from skellyforge.tools.viewer.synthetic_fit import apply_fit


def hand_metrics(captured, fit):
    """Audit unrounded world targets, bone spans and fitted landmark positions."""
    model, sequence = fit.model, fit.sequence
    rotations = np.asarray(sequence.quaternions)
    translations = np.asarray(sequence.translations)
    records = captured['records']
    errors, length_errors, segments = [], [], {}
    for b, name in enumerate(model['names']):
        if not any(part in name for part in ('carpal', 'phalanx')):
            continue
        definition = captured['skeleton'].segments[name].frame_definition
        start, end = definition.origin_point_name, definition.primary_point_name
        target_lengths = np.array([np.linalg.norm(r['points'][end] - r['points'][start]) for r in records])
        extent = model['display'][b][model['display_names'][b].index(end)]
        delta = np.abs(target_lengths - np.linalg.norm(extent))
        length_errors.extend(delta)
        rotation = Rotation.from_quat(rotations[:, b], scalar_first=True)
        segment_errors = []
        for key, point in zip(model['display_names'][b], model['display'][b], strict=True):
            predicted = translations[:, b] + rotation.apply(np.broadcast_to(point, (len(records), 3)))
            observed = np.array([r['points'][key] for r in records])
            segment_errors.extend(np.linalg.norm(predicted - observed, axis=1))
        errors.extend(segment_errors)
        segments[name] = dict(length_mm=float(np.linalg.norm(extent)),
            maximum_length_mismatch_mm=float(max(delta)),
            landmark_rms_mm=float(np.sqrt(np.mean(np.square(segment_errors)))))
    return dict(landmark_rms_mm=float(np.sqrt(np.mean(np.square(errors)))),
        maximum_landmark_error_mm=float(max(errors)),
        maximum_length_mismatch_mm=float(max(length_errors)), segments=segments)


def generate(output):
    captured = {}
    source = _build_data(root_motion=True, shoulders=True, elbows=True, head=True,
                         torso=True, legs=True, wrists=True, fingers=True, feet=True, capture=captured)
    options = human_fit_options()
    model = body_model(captured['skeleton'], captured['saved_model'], captured['segment_scales'],
                       options['shoulder_profile'], flexible_cervical=True)
    arguments, preparation = prepare_body_fit(model, captured['records'], **options)
    # Every synthetic landmark is a direct, present observation, including CMCs.
    for record in captured['records']:
        observed, _ = frame_targets(record, model)
        if sum(map(len, observed)) != len(captured['skeleton'].landmarks):
            raise ValueError('Synthetic landmark coverage is incomplete')
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, scale in [('current', arguments['angular_acceleration_scale']), ('weak-angular', 1e9)]:
        sequence = fit_prepared_human({**arguments, 'angular_acceleration_scale': scale}, model, captured['records'])
        fit = HumanFit(model, preparation, sequence)
        report[label] = {**hand_metrics(captured, fit), 'angular_acceleration_scale': scale}
        data = apply_fit(deepcopy(source), captured, fit)
        page = HTML_TEMPLATE.replace('__DATA__', json.dumps(data, allow_nan=False)).replace('__VENDORED_SCRIPTS__', _vendored_scripts())
        page = page.replace('var live = location.protocol', 'var live = false && location.protocol')
        page = page.replace('Independent segments &rarr; connected skeleton', f'Hand fit experiment: {label}')
        page = page.replace('connected (cyan)', f'{label} Ceres fit (cyan)')
        (output / f'{label}.html').write_text(page, encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'index.html').write_text('<h1>Synthetic hand fit audit</h1><p>Same observations and fixed lengths. Weak angular smoothing is an experiment, not a production fix.</p><p><a href="current.html">Current fit</a> | <a href="weak-angular.html">Weak angular smoothing</a> | <a href="report.json">Measurements</a></p>', encoding='utf-8')
    for label, metrics in report.items():
        print(f"{label}: hand RMS {metrics['landmark_rms_mm']:.3f} mm; maximum bone-length mismatch {metrics['maximum_length_mismatch_mm']:.3g} mm")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8776)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[2] / '.test-artifacts' / 'hand-fit-audit'
    generate(output)
    if args.serve:
        with ThreadingHTTPServer(('127.0.0.1', args.port), partial(SimpleHTTPRequestHandler, directory=str(output))) as server:
            print(f'Open http://127.0.0.1:{server.server_port}/ — Ctrl+C stops this experiment', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
