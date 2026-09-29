"""Compatibility entry point; implementation moved with its workflow."""
import sys
from diagnostics.generators import recording_fit_geometry as _implementation
sys.modules[__name__] = _implementation
