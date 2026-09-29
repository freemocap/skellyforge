"""Compatibility import for exploratory scripts; shared reader lives in tools."""
import sys
from skellyforge.tools import recording_data as _implementation
sys.modules[__name__] = _implementation
