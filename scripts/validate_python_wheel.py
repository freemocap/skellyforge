"""Check the distributable, including scientific definitions and offline assets.

Usage: python scripts/validate_python_wheel.py dist/skellyforge-*.whl
"""
import argparse
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile


def validate(path: Path) -> None:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        wheel = next(name for name in names if name.endswith('.dist-info/WHEEL'))
        headers = Parser().parsestr(archive.read(wheel).decode())
        if headers['Root-Is-Purelib'] != 'true' or headers.get_all('Tag') != ['py3-none-any']:
            raise ValueError('Expected a platform-independent pure Python wheel')
        forbidden = [name for name in names if
            name.lower().endswith(('.pyd', '.so', '.dll', '.dylib', '.cpp', '.h'))
            or '/skeleton/fitting/' in name or '/solver_inspector/' in name
            or '/fit_connected_' in name or '/_native' in name]
        if forbidden:
            raise ValueError(f'Unexpected native or connected-fitting payload: {forbidden}')
        required = {
            'skellyforge/definitions/human_skeleton/human_skeleton.yaml',
            'skellyforge/definitions/human_skeleton/rest_pose.yaml',
            'skellyforge/core/skeleton/pose/hydration.py',
            'skellyforge/core/skeleton/pose/model_scale_fitting.py',
            'skellyforge/core/trajectories/__init__.py',
            'skellyforge/tools/viewer/web/vendor/three.min.js',
            'skellyforge/tools/viewer/web/vendor/OrbitControls.js',
            'skellyforge/tools/viewer/web/viewer_geometry.js',
        }
        if missing := required - names:
            raise ValueError(f'Missing package data: {sorted(missing)}')
    print(f'{path.name}: pure Python; required definitions, geometry and viewer assets present')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    validate(parser.parse_args().wheel)
