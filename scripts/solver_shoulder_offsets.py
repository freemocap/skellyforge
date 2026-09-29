"""Compatibility entry point; implementation moved with its workflow."""
import sys
from experiments.generators import solver_shoulder_offsets as _implementation
sys.modules[__name__] = _implementation
