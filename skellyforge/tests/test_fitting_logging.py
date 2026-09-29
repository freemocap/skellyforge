"""Logging preserves application ownership and native fitting results."""
import logging
import subprocess
import sys
import io
import re
from types import SimpleNamespace
import pytest

import numpy as np

from skellyforge.tests.test_native_window_sequence import seeded
from skellyforge.core.skeleton.fitting.window_sequence import fit_windows
from skellyforge.core.skeleton.fitting.terminal_progress import TerminalProgress, COLUMNS
from skellylogs.formatters.color_formatter import LOG_COLOR_CODES


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
    assert not caplog.records





def reporter(stream, clock):
    return TerminalProgress(frames=12, segments=5, total=10, active=3, boundary=2,
        iterations=200, tolerance=1e-6, stream=stream, clock=clock)


def row(index=0, converged=True):
    return dict(index=index, active_start=index, active_end=index+2, fixed_start=max(0,index-2),
        iterations=4, initial_cost=100., final_cost=25., seconds=.01, wall_seconds=.012,
        parameter_blocks=30, residual_blocks=60, converged=converged)


def test_redirected_table_has_definitions_single_rows_and_no_ansi():
    output = io.StringIO()
    terminal = reporter(output, lambda: 0.)
    terminal.update(row(), unconverged=0)
    terminal.update(row(3), unconverged=0)
    terminal.update(row(4, False), unconverged=1)
    text = output.getvalue()
    for name, _, description in COLUMNS:
        assert description in text
    assert '\033' not in text and '\r' not in text
    rows = [line for line in text.splitlines() if re.match(r'\s+\d+\s+\d+-\d+', line)]
    assert len(rows) == 3
    assert rows[0].endswith('CONVERGED') and rows[2].endswith('USABLE')
    assert '75.0' in rows[0]


def test_live_eta_colors_and_cleanup(monkeypatch):
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm')
    class Terminal(io.StringIO):
        def isatty(self): return True
    output = Terminal()
    times = iter([0., 2., 6.])
    terminal = reporter(output, lambda: next(times))
    terminal.update(row(), unconverged=0)
    terminal.update(row(1, False), unconverged=1)
    terminal.line('Evaluating full sequence', 'INFO')
    text = output.getvalue()
    assert 'Mean (ms/window): 3000.0 | ETA (s): 24' in text
    assert LOG_COLOR_CODES['INFO'] in text
    assert LOG_COLOR_CODES['SUCCESS'] in text
    assert LOG_COLOR_CODES['WARNING'] in text
    assert terminal.live_width == 0


def test_closed_terminal_does_not_abort_fit():
    output = io.StringIO()
    terminal = reporter(output, lambda: 0.)
    output.close()
    terminal.update(row(), unconverged=0)
    assert not terminal.enabled


def test_native_verbose_diagnostics_are_disabled(capfd):
    arguments, _ = seeded()
    fit_windows(arguments)
    output = capfd.readouterr()
    for text in (output.out, output.err):
        assert 'Allocating values array' not in text
        assert 'Symbolic Analysis' not in text


def test_failure_remains_immediate_and_callback_cancellation_is_preserved(monkeypatch, capsys):
    from skellyforge.core.skeleton.fitting import window_sequence
    arguments, _ = seeded()
    with pytest.raises(RuntimeError, match='cancelled by caller'):
        fit_windows(arguments, progress=lambda *_: (_ for _ in ()).throw(RuntimeError('cancelled by caller')))
    monkeypatch.setattr(window_sequence._native, 'fit_chain_sequence',
        lambda **_: SimpleNamespace(usable=False, report='deliberate unusable result'))
    with pytest.raises(RuntimeError, match='deliberate unusable result'):
        fit_windows(arguments)
    assert 'FAILED: window 1/' in capsys.readouterr().err


def test_live_mean_keeps_only_twenty_intervals(monkeypatch):
    monkeypatch.setenv('NO_COLOR', '1')
    monkeypatch.setenv('TERM', 'xterm')
    class Terminal(io.StringIO):
        def isatty(self): return True
    output = Terminal()
    times = iter([0., 100., *range(101, 121)])
    terminal = reporter(output, lambda: next(times))
    terminal.total = 30
    for index in range(21): terminal.update(row(index), unconverged=0)
    assert 'Mean (ms/window): 1000.0 | ETA (s): 9' in output.getvalue()
    assert '\033' not in output.getvalue()
