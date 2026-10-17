"""Compatibility entry point; implementation lives in the supported viewer package."""
from skellyforge.tools.viewer import synthetic as _implementation
# Preserve diagnostic imports, including private helpers used by existing tests.
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith('__')})
if __name__ == '__main__':
    main()
