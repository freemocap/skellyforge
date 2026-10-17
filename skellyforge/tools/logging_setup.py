"""Standalone process logging; importing Forge does not alter host streams."""
import sys
from skellylogs import LogLevels, configure_logging


def configure_standalone_logging():
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    configure_logging(LogLevels.INFO, use_websocket=False)
