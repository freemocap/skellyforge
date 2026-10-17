"""Keep test scratch files isolated from other Windows users and agent sessions."""
from pathlib import Path
from tempfile import mkdtemp


def pytest_configure(config):
    if config.option.basetemp is not None:
        return
    scratch = Path(__file__).resolve().parent / '.test-artifacts' / 'pytest'
    scratch.mkdir(parents=True, exist_ok=True)
    # Reserve a fresh directory: pytest may clear basetemp before using it.
    # Never point it at the shared parent or another run's files.
    config.option.basetemp = mkdtemp(prefix='run-', dir=scratch)
