"""Serve previously generated experiment output without running any fits."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .workspace import OUTPUT_FOLDER


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?', 1)[0] == '/':
            self.send_response(302)
            self.send_header('Location', '/solver_viewer.html')
            self.end_headers()
            return
        super().do_GET()


def serve(folder: Path, port: int):
    if not (folder / 'solver_viewer.html').is_file():
        raise FileNotFoundError('Generate the viewer first: uv run poe experiment-solver-viewer --serve')
    with ThreadingHTTPServer(('127.0.0.1', port), partial(ViewerHandler, directory=str(folder))) as server:
        print(f'Open http://127.0.0.1:{server.server_port}/solver_viewer.html — Ctrl+C stops the server', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args()
    serve(OUTPUT_FOLDER, args.port)


if __name__ == '__main__':
    main()
