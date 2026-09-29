"""Compatibility entry point for the packaged comparison renderer."""
import sys
from skellyforge.tools.viewer import comparison as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
