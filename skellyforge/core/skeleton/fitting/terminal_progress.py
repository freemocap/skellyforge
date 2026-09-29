"""Terminal-only fit diagnostics. No handlers, logging records or solver state."""
from collections import deque
import os
import sys
from time import perf_counter

import numpy as np
from skellylogs.formatters.color_formatter import LOG_COLOR_CODES




COLUMNS = (
    ('Window', 6, 'Completed solve-window number, starting at 1.'),
    ('Active', 11, 'Inclusive frame indices adjusted by this solve, starting at 0.'),
    ('Fixed', 11, 'Inclusive preceding frame indices held unchanged; - means none.'),
    ('Iterations', 10, 'Iteration count reported by Ceres for this window.'),
    ('Initial cost', 13, 'Weighted objective before this solve; different windows have different objectives.'),
    ('Final cost', 13, 'The same window objective after solving.'),
    ('Reduction (%)', 13, '100 * (initial - final) / initial; - if initial cost is zero.'),
    ('Native (ms)', 12, 'Duration reported by the native solver.'),
    ('Call (ms)', 11, 'Python wall time around the native call, including argument/result transfer.'),
    ('Extra (ms)', 11, 'Call minus native duration; excludes Python work outside that call.'),
    ('Param blocks', 12, 'Ceres parameter groups, not scalar unknowns.'),
    ('Resid blocks', 12, 'Ceres residual groups, not scalar errors.'),
    ('Status', 11, 'CONVERGED or USABLE (accepted without convergence). Failures are logged immediately.'),
)


class TerminalProgress:
    def __init__(self, *, frames, segments, total, active, boundary, iterations, tolerance,
                 stream=None, clock=perf_counter):
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.started = self.previous = clock()
        self.intervals = deque(maxlen=20)
        self.total = total
        self.enabled = self.stream is not None
        self.interactive = self.enabled and self.stream.isatty() and os.environ.get('TERM') != 'dumb'
        self.color = self.interactive and 'NO_COLOR' not in os.environ
        self.live_width = 0
        self.line('SKELLYFORGE - CONNECTED SKELETON FIT', 'INFO')
        self.line(f'Recording: {frames} frames | {segments} segments | {total} windows')
        self.line(f'Settings: {active} adjustable frames + up to {boundary} fixed preceding frames | max iterations: {iterations} | function tolerance: {tolerance:g}')
        self.line('\nCOLUMN DEFINITIONS', 'INFO')
        for name, _, description in COLUMNS:
            self.line(f'{name:<18} {description}')
        self.line('\nLIVE STATUS DEFINITIONS', 'INFO')
        for name, description in (
            ('Progress', 'Completed windows / total windows, also shown as a percentage.'),
            ('Elapsed (s)', 'Wall-clock seconds since sequence solving began.'),
            ('Mean (ms/window)', 'Mean completion interval over the last 20 windows, or all completed windows if fewer.'),
            ('ETA (s)', 'Remaining windows times that mean; excludes subsequent full-sequence evaluation.'),
            ('Unconverged', 'Completed usable windows that did not converge.'),
        ):
            self.line(f'{name:<18} {description}')
        self.line('')
        self.line(' '.join(name.rjust(width) for name, width, _ in COLUMNS), 'INFO')

    def paint(self, text, level):
        return f'{LOG_COLOR_CODES[level]}{text}\033[0m' if self.color and level else text

    def write(self, text):
        if self.enabled:
            try:
                self.stream.write(text)
                self.stream.flush()
            except (OSError, ValueError):
                # A closed terminal must not discard an expensive numerical result.
                self.enabled = False

    def clear(self):
        if self.live_width:
            self.write('\r' + ' ' * self.live_width + '\r')
            self.live_width = 0

    def line(self, text, level=None):
        self.clear()
        self.write(self.paint(text, level) + '\n')

    def update(self, row, *, unconverged):
        now = self.clock()
        self.intervals.append(now - self.previous)
        self.previous = now
        done = row['index'] + 1
        initial, final = row['initial_cost'], row['final_cost']
        values = [str(done), f"{row['active_start']}-{row['active_end']}",
            f"{row['fixed_start']}-{row['active_start']-1}" if row['active_start'] > row['fixed_start'] else '-',
            str(row['iterations']), f'{initial:.6g}', f'{final:.6g}',
            f'{100*(initial-final)/initial:.1f}' if initial else '-',
            f"{row['seconds']*1000:.1f}", f"{row['wall_seconds']*1000:.1f}",
            f"{(row['wall_seconds']-row['seconds'])*1000:.1f}",
            str(row['parameter_blocks']), str(row['residual_blocks']),
            'CONVERGED' if row['converged'] else 'USABLE']
        cells = [value.rjust(width) for value, (_, width, _) in zip(values, COLUMNS)]
        cells[-1] = self.paint(cells[-1], 'SUCCESS' if row['converged'] else 'WARNING')
        self.line(' '.join(cells))
        if self.interactive:
            self.clear()
            mean = sum(self.intervals) / len(self.intervals)
            status = (f'Progress {done}/{self.total} ({100*done/self.total:.1f}%) | Elapsed (s): {now-self.started:.1f}'
                f' | Mean (ms/window): {1000*mean:.1f} | ETA (s): {(self.total-done)*mean:.0f} | Unconverged: {unconverged}')
            self.write(self.paint(status, 'INFO'))
            self.live_width = len(status)

    def finish(self, trace, evaluation_seconds):
        self.line('FIT COMPLETE', 'SUCCESS' if all(row['converged'] for row in trace) else 'WARNING')
        for field, label, factor in [('iterations', 'Iterations', 1), ('seconds', 'Native solve (ms)', 1000)]:
            values = np.asarray([row[field] for row in trace]) * factor
            self.line(f'{label}: median {np.median(values):.1f} | p95 {np.percentile(values,95):.1f} | max {max(values):.1f}')
        slowest = max(trace, key=lambda row: row['wall_seconds'])
        self.line(f"Converged: {sum(row['converged'] for row in trace)}/{self.total} | Slowest window: {slowest['index']+1} | Window native total (s): {sum(row['seconds'] for row in trace):.3f} | Evaluation wall (s): {evaluation_seconds:.3f} | Total wall (s): {self.clock()-self.started:.3f}")
