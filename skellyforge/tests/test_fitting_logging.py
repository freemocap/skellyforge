"""Logging preserves application ownership and native fitting results."""
import logging
import subprocess
import sys

import numpy as np

from skellyforge.tests.test_native_window_sequence import seeded
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows


def test_import_preserves_application_handlers():
    subprocess.run([sys.executable, '-c', "import logging; h=logging.StreamHandler(); root=logging.getLogger(); root.addHandler(h); root.setLevel(logging.ERROR); import skellyforge; assert root.handlers == [h]; assert root.level == logging.ERROR"], check=True)


def test_progress_logging_does_not_change_fit(caplog):
    arguments, _ = seeded()
    baseline = fit_windows(arguments)
    callbacks = []
    with caplog.at_level(logging.DEBUG, logger='skellyforge.core.skeleton.fitting.window_sequence'):
        result = fit_windows(arguments, progress=lambda row, total: callbacks.append((row, total)))
    np.testing.assert_array_equal(result.quaternions, baseline.quaternions)
    np.testing.assert_array_equal(result.translations, baseline.translations)
    np.testing.assert_array_equal(result.lengths, baseline.lengths)
    assert len(callbacks) == len(result.processing['windows'])
    assert 'Starting Ceres sequence' in caplog.text
    assert 'Ceres progress' in caplog.text
    assert 'no optimization' in caplog.text
    assert 'Ceres sequence finished' in caplog.text
