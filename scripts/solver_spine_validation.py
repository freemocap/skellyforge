"""Compatibility entry point; implementation moved with its workflow."""
import sys
from diagnostics.generators import solver_spine_validation as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
