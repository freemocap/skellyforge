"""Use the same source assets in development and install them unchanged in wheels."""
from pathlib import Path


def asset_directory():
    installed = Path(__file__).with_name('web')
    if installed.is_dir():
        return installed
    raise FileNotFoundError('Viewer assets missing from SkellyForge installation')


def vendored_scripts():
    return '\n'.join((asset_directory() / 'vendor' / name).read_text(encoding='utf-8')
                     for name in ('three.min.js', 'OrbitControls.js'))


def geometry_script():
    return (asset_directory() / 'viewer_geometry.js').read_text(encoding='utf-8')
