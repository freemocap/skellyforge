"""Local, read-only review server; synthetic hydration is the only recomputation."""
import hashlib
import json
import logging
import mimetypes
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote

from .assets import asset_directory
from .comparison import comparison_data
from . import synthetic

logger = logging.getLogger(__name__)


def contained_file(root, relative):
    root = root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(relative)
    return target


class Viewer:
    def __init__(self, results, captures, media):
        self.results, self.captures, self.media = map(Path, (results, captures, media))
        self.pages = {}
        self.status = {}
        try:
            from .synthetic_fit import source_hashes
            self.synthetic_fit = json.loads((self.results/'synthetic.json').read_text(encoding='utf-8'))
            if self.synthetic_fit.get('fit_source_hashes') != source_hashes():
                raise ValueError('Synthetic fit is stale; run viewer --prepare-synthetic-fit')
            self.status['synthetic-fit'] = dict(available=True, frames=self.synthetic_fit['frame_count'], source='Known standard-human synthetic landmarks', fit=str(self.results/'synthetic.json'))
        except (OSError, ValueError) as error:
            self.synthetic_fit = None
            self.status['synthetic-fit'] = dict(available=False, error=str(error))
        for name, folder in [('test', self.results / 'test'), ('sample', self.results)]:
            path = folder / 'candidate.json'
            try:
                candidate = json.loads(path.read_text(encoding='utf-8'))
                source = candidate['metadata']['recording']
                parquet = Path(source['path'])
                if not parquet.is_file():
                    raise FileNotFoundError(f"Prepared Parquet missing: {parquet}")
                with parquet.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != source['sha256']:
                    raise ValueError('Saved fit uses a different Parquet revision; regenerate it')
                data = comparison_data([candidate])
                page = (asset_directory() / 'recording_comparison.html.template').read_text(encoding='utf-8')
                page = page.replace('__ALTERNATE_VIEW__', '')
                page = page.replace('__DATA__', json.dumps(data, allow_nan=False).replace('<', '\\u003c'))
                self.pages[name] = page.encode('utf-8')
                self.status[name] = dict(available=True, frames=len(data['times']), source=str(parquet), fit=str(path))
                logger.info('Loaded %s recording review: %d frames, source=%s', name, len(data['times']), parquet)
            except (OSError, ValueError, KeyError) as error:
                self.status[name] = dict(available=False, error=str(error), fit=str(path))
                logger.warning('%s recording review unavailable: %s', name, error)

    def synthetic_page(self):
        data = synthetic._build_data()
        return synthetic.HTML_TEMPLATE.replace('__DATA__', json.dumps(data, allow_nan=False)).replace(
            '__VENDORED_SCRIPTS__', synthetic._vendored_scripts()).encode('utf-8')


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, viewer, **kwargs):
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def send(self, content, content_type='text/html; charset=utf-8', status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/':
                return self.send(Path(__file__).with_name('index.html').read_bytes())
            if path == '/status':
                return self.send(json.dumps(self.viewer.status).encode('utf-8'), 'application/json')
            if path == '/synthetic-fit':
                if self.viewer.synthetic_fit is None:
                    raise FileNotFoundError(self.viewer.status['synthetic-fit']['error'])
                data = self.viewer.synthetic_fit
                page = synthetic.HTML_TEMPLATE.replace('__DATA__', json.dumps(data, allow_nan=False)).replace('__VENDORED_SCRIPTS__', synthetic._vendored_scripts())
                page = page.replace('var live = location.protocol', 'var live = false && location.protocol')
                page = page.replace('connected (cyan)', 'Ceres fit (cyan)')
                page = page.replace('Independent segments &rarr; connected skeleton', 'Synthetic segments &rarr; accepted Ceres fit')
                page = page.replace('Interactive inputs need Python: run scripts/generate_skeleton_viewer.py --serve and open the printed local URL. This file is a snapshot.', 'Saved Ceres fit of all-motion synthetic landmarks. Input controls are disabled; use the Synthetic humanoid tab for live hydration experiments.')
                return self.send(page.encode('utf-8'))
            if path == '/.solver_inspections/synthetic/manifest.json':
                return self.send(json.dumps(dict(windows=[dict(index=0, file='window_0.json', frame_start=0, frame_end=2)])).encode('utf-8'), 'application/json')
            if path == '/.solver_inspections/synthetic/window_0.json':
                return self.send((self.viewer.results / 'synthetic_window_0.json').read_bytes(), 'application/json')
            if path == '/synthetic':
                return self.send(self.viewer.synthetic_page())
            if path in ('/test', '/sample', '/recording_test_spine_positions.html'):
                name = 'test' if path.endswith('.html') else path[1:]
                if name not in self.viewer.pages:
                    return self.send(self.viewer.status[name]['error'].encode('utf-8'), 'text/plain; charset=utf-8', 404)
                return self.send(self.viewer.pages[name])
            if path.startswith('/.solver_inspections/'):
                target = contained_file(self.viewer.captures, path.removeprefix('/.solver_inspections/'))
                if target.suffix != '.json':
                    raise FileNotFoundError(path)
            elif path.startswith('/.solver_media/'):
                target = contained_file(self.viewer.media, path.removeprefix('/.solver_media/'))
                if target.suffix != '.jpg':
                    raise FileNotFoundError(path)
            else:
                target = contained_file(asset_directory(), path.lstrip('/'))
                if target.suffix not in ('.js', '.css', '.html'):
                    raise FileNotFoundError(path)
            return self.send(target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        except (OSError, ValueError) as error:
            self.send(str(error).encode('utf-8'), 'text/plain; charset=utf-8', 404)

    def do_POST(self):
        if self.path != '/data':
            return self.send(b'Not found', 'text/plain', 404)
        try:
            if synthetic._source_hashes() != synthetic.LOADED_SOURCE_HASHES:
                return self.send(b'Source changed; restart viewer', 'text/plain', 409)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2048:
                raise ValueError('Invalid request size')
            values = json.loads(self.rfile.read(length))
            switches = {'root_motion', 'shoulders', 'elbows', 'head', 'spread_hands'}
            if not isinstance(values, dict) or set(values) != switches | {'noise_mm'}:
                raise ValueError('Expected synthetic motion switches and noise_mm')
            if any(type(values[key]) is not bool for key in switches):
                raise ValueError('Motion switches must be booleans')
            if type(values['noise_mm']) not in (float, int) or not 0 <= values['noise_mm'] <= 20:
                raise ValueError('Noise must be between 0 and 20 mm')
            data = synthetic._build_data(**values)
            self.send(json.dumps(data, allow_nan=False).encode('utf-8'), 'application/json')
        except (ValueError, TypeError) as error:
            self.send(str(error).encode('utf-8'), 'text/plain; charset=utf-8', 400)
        except Exception:
            logger.exception('Synthetic hydration failed')
            self.send(b'Synthetic hydration failed; see log', 'text/plain', 500)

    def log_message(self, format, *args):
        logger.debug(format, *args)


def serve(viewer, port):
    with ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, viewer=viewer)) as server:
        logger.info('SkellyForge viewer: http://127.0.0.1:%d', server.server_port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
