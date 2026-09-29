"""Controlled dropouts: raw keypoints -> gap filling -> connected sequence fitting."""
import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from skellyforge import _native
from skellyforge.core.skeleton.fitting import fit_windows
from skellyforge.core.trajectories import fill_trajectory_gaps
from test_support.chain import chain_inputs


CASES = {
    'single_point': 'One keypoint disappears briefly',
    'whole_segment': 'All keypoints on the middle segment disappear',
    'absence': 'Whole model exits and re-enters',
    'ends': 'Missing recording ends and one keypoint’s ends',
    'never_seen': 'Distal segment never observed',
}


def experiment(kind):
    local, parent, child, times, _, records = chain_inputs(noise=1., gap=0)
    truth = np.array([r['truth'] for r in records]).reshape(len(times), 24, 3)
    raw = np.array([r['observed'] for r in records]).reshape(len(times), 24, 3)
    if kind == 'single_point': raw[15:23, 8] = np.nan
    elif kind == 'whole_segment': raw[15:23, 8:16] = np.nan
    elif kind == 'absence': raw[17:25] = np.nan
    elif kind == 'ends':
        raw[:4] = raw[-4:] = np.nan
        raw[:10, 8] = raw[-10:, 8] = np.nan
    elif kind == 'never_seen': raw[:, 16:] = np.nan
    else: raise ValueError(kind)
    filled, report = fill_trajectory_gaps(points=raw, timestamps_s=times)
    fitted = np.full_like(raw, np.nan)
    runs = []
    for start, stop in report.active_spans:
        if stop - start < 3:
            runs.append(dict(start=start, stop=stop, skipped='Fewer than three frames'))
            continue
        observed, indices, initial, roots = [], [], [], []
        for frame in filled[start:stop].reshape(-1, 3, 8, 3):
            slots = [np.flatnonzero(np.isfinite(body).all(axis=-1)).tolist() for body in frame]
            targets = [body[slot].tolist() for body, slot in zip(frame, slots)]
            rotations = []
            for b, slot in enumerate(slots):
                if len(slot) >= 3:
                    seed = _native.fit_rigid(local=local[slot].tolist(), observed=targets[b],
                        quaternion=[1., 0., 0., 0.], translation=[0., 0., 0.])
                    rotations.append(seed.quaternion)
                    if b == 0: root = seed.translation
                else:
                    if b == 0: raise ValueError('This fixture requires root initialization evidence')
                    rotations.append(rotations[b - 1])
            observed.append(targets); indices.append(slots); initial.append(rotations); roots.append(root)
        result = fit_windows(dict(local=[local.tolist()] * 3, observed=observed,
            observation_indices=indices, parent_attachments=parent.tolist(), child_attachments=child.tolist(),
            parent_indices=[0, 1], times=times[start:stop].tolist(), position_scale=10.,
            linear_acceleration_scale=3000., angular_acceleration_scale=20.,
            initial_quaternions=initial, initial_roots=roots,
            rest_relative_quaternions=[[1., 0., 0., 0.]] * 2, rest_pose_scale=1.))
        for i in range(stop - start):
            fitted[start + i] = np.array([Rotation.from_quat(result.quaternions[i][b], scalar_first=True).apply(local)
                + result.translations[i][b] for b in range(3)]).reshape(24, 3)
        runs.append(dict(start=start, stop=stop, converged=bool(result.converged)))
    measured = report.measured_support(filled)
    predicted = np.isfinite(fitted).all(axis=-1)
    error = np.linalg.norm(fitted - truth, axis=-1)
    filled_mask = np.isfinite(filled).all(axis=-1) & ~measured
    def rms(mask):
        return float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None
    return dict(name=CASES[kind], times=times.tolist(), raw=raw, filled=filled, fitted=fitted,
        truth=truth, measured=measured.tolist(), report=report.to_dict(), runs=runs,
        rms_mm=rms(predicted), inferred_rms_mm=rms(predicted & ~measured),
        filled_count=int(filled_mask.sum()), blank_count=int((~predicted.any(axis=1)).sum()))


def json_ready(value):
    if isinstance(value, np.ndarray): return json_ready(value.tolist())
    if isinstance(value, dict): return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8777)
    args = parser.parse_args()
    cases = [experiment(kind) for kind in CASES]
    folder = Path(__file__).resolve().parents[2] / '.test-artifacts' / 'viewers' / 'partial_observations'
    folder.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name('partial_observations.html.template').read_text(encoding='utf-8')
    page = folder / 'index.html'
    page.write_text(template.replace('__DATA__', json.dumps(json_ready(cases), allow_nan=False)), encoding='utf-8')
    print(f'Viewer saved to {page}', flush=True)
    if args.serve:
        with ThreadingHTTPServer(('127.0.0.1', args.port), partial(SimpleHTTPRequestHandler, directory=str(folder))) as server:
            print(f'Open http://127.0.0.1:{server.server_port}/ — Ctrl+C stops the server', flush=True)
            try: server.serve_forever()
            except KeyboardInterrupt: pass


if __name__ == '__main__':
    main()
