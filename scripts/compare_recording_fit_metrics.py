"""Compatibility entry point; implementation moved with its workflow."""
import sys
from diagnostics.generators import compare_recording_fit_metrics as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
