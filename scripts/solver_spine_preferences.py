"""Compatibility entry point; implementation moved with its workflow."""
import sys
from experiments.generators import solver_spine_preferences as _implementation
sys.modules[__name__] = _implementation
