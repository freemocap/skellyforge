"""Renderers use one asset source and keep generated files separate."""
import importlib
from pathlib import Path

from skellyforge.tools.viewer import workspace, comparison
from skellyforge.tools.viewer.assets import asset_directory
from experiments.generators import generate_solver_viewer


def test_saved_viewer_root_opens_the_generated_page(tmp_path):
    from functools import partial
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import urlopen
    from skellyforge.tools.viewer.serve_saved import ViewerHandler

    (tmp_path / 'solver_viewer.html').write_text('saved experiment output', encoding='utf-8')
    with ThreadingHTTPServer(('127.0.0.1', 0), partial(ViewerHandler, directory=str(tmp_path))) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/') as response:
                assert response.url.endswith('/solver_viewer.html')
                assert response.read() == b'saved experiment output'
        finally:
            server.shutdown()
            thread.join()


def test_renderers_use_relocated_assets(tmp_path, monkeypatch):
    output = tmp_path / 'pages'
    monkeypatch.setattr(workspace, 'OUTPUT_FOLDER', output)
    monkeypatch.setattr(generate_solver_viewer, 'OUTPUT_FOLDER', output)
    monkeypatch.setattr(comparison, 'OUTPUT_FOLDER', output)
    generate_solver_viewer.render_experiments(bank=[])
    solver = (output / 'solver_viewer.html').read_text(encoding='utf-8')
    assert 'const EXPERIMENTS=[]' in solver
    assert '__STYLE__' not in solver and '__VIEWER__' not in solver and '__ASSETS__' not in solver
    comparison.write_comparison(dict(solutions=[], times=[]))
    page = (output / 'recording_comparison.html').read_text(encoding='utf-8')
    assert 'recording_comparison.js?v=' in page
    assert (output / 'recording_comparison.js').read_bytes() == (asset_directory() / 'recording_comparison.js').read_bytes()
    assert (output / 'vendor/three.min.js').is_file()


def test_legacy_generator_imports_resolve_to_canonical_modules():
    root = Path(__file__).resolve().parents[2]
    for category in ('diagnostics', 'experiments'):
        for path in (root / category / 'generators').glob('*.py'):
            if path.name == '__init__.py':
                continue
            module = importlib.import_module(f'{category}.generators.{path.stem}')
            assert importlib.import_module(f'scripts.{path.stem}') is module
