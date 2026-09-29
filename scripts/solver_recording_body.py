"""Compatibility entry point; implementation moved with its workflow."""
import sys
from experiments.generators import solver_recording_body as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
