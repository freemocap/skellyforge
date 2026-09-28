"""Supported viewer routing and source ownership, without browser mocks of math."""
import json
from pathlib import Path
from threading import Thread
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
from urllib.error import HTTPError
import pytest
from skellyforge.tools.viewer.server import Viewer, Handler, contained_file
from skellyforge.tools.viewer import synthetic


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
