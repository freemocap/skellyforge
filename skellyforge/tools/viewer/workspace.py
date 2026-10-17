"""Local generated viewer output, separate from source assets and saved recordings."""
from pathlib import Path
import shutil

from .assets import asset_directory

_checkout = Path(__file__).resolve().parents[3]
REPO_ROOT = _checkout if (_checkout / 'pyproject.toml').is_file() else Path.cwd()
OUTPUT_FOLDER = REPO_ROOT / '.test-artifacts' / 'viewers'


def prepare_output():
    """Copy viewer support files beside generated pages, without copying old results."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    shutil.copytree(asset_directory(), OUTPUT_FOLDER, dirs_exist_ok=True)
    return OUTPUT_FOLDER
