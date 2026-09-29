"""Supported viewer routing and source ownership, without browser mocks of math."""
import json
from pathlib import Path
from threading import Thread
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
from urllib.error import HTTPError
import pytest
import numpy as np
from skellyforge.tools.viewer.server import Viewer, Handler, contained_file
from skellyforge.tools.viewer import synthetic


def test_joint_motion_and_region_freezing():
    from scipy.spatial.transform import Rotation
    active, frozen = {}, {}
    settings = dict(legs=True, wrists=True, fingers=True, feet=True)
    synthetic._build_data(**settings, capture=active)
    regions = dict.fromkeys(synthetic.REGIONS, True)
    regions.update(left_hand=False, left_foot=False, left_leg=False)
    synthetic._build_data(**settings, regions=regions, capture=frozen)
    for name in ('left_carpals', 'left_index_proximal_phalanx', 'left_foot', 'left_toes', 'left_upper_leg'):
        moving = np.array([r['rotations'][name] for r in active['records']])
        assert np.max(np.ptp(moving, axis=0)) > .02, name
        fixed = np.array([r['rotations'][name] for r in frozen['records']])
        np.testing.assert_allclose(fixed, np.broadcast_to(fixed[0], fixed.shape), atol=1e-9)
    for name in ('right_carpals', 'right_index_proximal_phalanx', 'right_foot', 'right_upper_leg'):
        moving = np.array([r['rotations'][name] for r in frozen['records']])
        assert np.max(np.ptp(moving, axis=0)) > .02, name
    for child, parent, axes in [('left_upper_leg', 'pelvis', (0, 1)), ('left_foot', 'left_lower_leg', (0, 2))]:
        records = active['records']
        q = Rotation.from_quat([r['rotations'][child] for r in records], scalar_first=True)
        p = Rotation.from_quat([r['rotations'][parent] for r in records], scalar_first=True)
        relative = p.inv() * q
        change = (relative * relative[0].inv()).as_rotvec()
        for axis in axes:
            assert np.ptp(change[:, axis]) > .1


def test_synthetic_hand_targets_match_fixed_solver_lengths():
    """CMC offsets must not be counted again as metacarpal length."""
    from skellyforge.core.skeleton.fitting.body_model import body_model, frame_targets
    captured = {}
    synthetic._build_data(shoulders=True, elbows=True, torso=True, unequal_scales=True, capture=captured)
    model = body_model(captured['skeleton'], captured['saved_model'], captured['segment_scales'])
    for record in captured['records']:
        observed, _ = frame_targets(record, model)
        assert sum(map(len, observed)) == len(captured['skeleton'].landmarks)
        for b, name in enumerate(model['names']):
            if not any(part in name for part in ('carpal', 'phalanx')):
                continue
            definition = captured['skeleton'].segments[name].frame_definition
            start, end = definition.origin_point_name, definition.primary_point_name
            target_length = np.linalg.norm(record['points'][end] - record['points'][start])
            local = model['display'][b][model['display_names'][b].index(end)]
            assert target_length == pytest.approx(np.linalg.norm(local), abs=1e-9), name


def test_torso_and_leg_motion_reaches_synthetic_observations():
    stationary, moving = {}, {}
    synthetic._build_data(capture=stationary)
    data = synthetic._build_data(torso=True, legs=True, capture=moving)
    assert data['motion']['torso'] and data['motion']['legs']
    for name in ('thoracic', 'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg'):
        fixed = np.asarray([frame['rotations'][name] for frame in stationary['records']])
        rotations = np.asarray([frame['rotations'][name] for frame in moving['records']])
        np.testing.assert_allclose(fixed, np.broadcast_to(fixed[0], fixed.shape), atol=1e-10)
        assert np.isfinite(rotations).all()
        assert np.max(np.ptp(rotations, axis=0)) > .01, name
    assert len(moving['records']) == data['frame_count']


def test_synthetic_capture_uses_the_displayed_inputs():
    capture = {}
    data = synthetic._build_data(shoulders=True, capture=capture)
    assert len(capture['records']) == data['frame_count']
    assert capture['skeleton'].name
    for record, frame in zip(capture['records'], data['frames'], strict=True):
        for name, position in zip(data['landmarks_meta'], frame['landmarks'], strict=True):
            import numpy as np
            np.testing.assert_allclose(record['points'][name], position, atol=0.0051, rtol=0  # Existing viewer rounds display coordinates to 0.01 mm.
            )
            np.testing.assert_array_equal(record['keypoints']['synthetic:'+name], record['points'][name])


def test_server_reports_missing_fits_and_restricts_file_roots(tmp_path):
    viewer = Viewer(tmp_path, tmp_path, tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, viewer=viewer))
    thread = Thread(target=server.serve_forever)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        status = json.loads(urlopen(base+'/status').read())
        assert not status['test']['available']
        assert b'Pipeline viewer' in urlopen(base+'/').read()
        assert b'Motion review' in urlopen(base+'/simple').read()
        assert b'fittedEndpoint' in urlopen(base+'/simple.js').read()
        assert b'--video-width' in urlopen(base+'/simple.css').read()
        assert b'createComparisonScene' in urlopen(base+'/recording_comparison_scene.js').read()
        with pytest.raises(HTTPError) as error:
            urlopen(base+'/test')
        assert error.value.code == 404
        with pytest.raises(HTTPError):
            urlopen(base+'/../pyproject.toml')
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_containment_rejects_parent_access(tmp_path):
    with pytest.raises(FileNotFoundError):
        contained_file(tmp_path, '../pyproject.toml')


def test_standalone_logging_uses_utf8_with_legacy_redirected_streams(tmp_path):
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "from skellyforge.tools.logging_setup import configure_standalone_logging; configure_standalone_logging(); import sys, logging; assert sys.stdout.encoding == 'utf-8'; assert sys.stderr.encoding == 'utf-8'; logging.getLogger('skellyforge').info('UTF8 test')"],
        env={**os.environ, 'PYTHONIOENCODING': 'cp1252', 'SKELLYLOGS_LOG_DIR': str(tmp_path)},
        capture_output=True, check=True)
    assert b'UnicodeEncodeError' not in result.stderr
    assert b'UTF8 test' in result.stdout + result.stderr
