"""Compatibility entry point; implementation moved with its workflow."""
import sys
from diagnostics.generators import generate_real_skeleton_viewer as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
