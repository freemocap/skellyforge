"""Supported viewer entry point. Does not run or modify recording pipelines."""
import argparse
from pathlib import Path
from skellyforge.tools.logging_setup import configure_standalone_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--results', type=Path, default=Path('build/spine_positions'))
    parser.add_argument('--captures', type=Path, default=Path('scripts/.solver_inspections'))
    parser.add_argument('--media', type=Path, default=Path('scripts/.solver_media'))
    parser.add_argument('--prepare-synthetic-fit', action='store_true', help='Generate the 60-frame all-motion fixture through fit_human before serving')
    args = parser.parse_args()
    configure_standalone_logging()
    if args.prepare_synthetic_fit:
        from .synthetic_fit import prepare
        prepare(args.results)
    from .server import Viewer, serve
    serve(Viewer(args.results, args.captures, args.media), args.port)


if __name__ == '__main__':
    main()
