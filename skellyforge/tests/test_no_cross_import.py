"""Enforce the runtime boundary: skellyforge must never import skellytracker or freemocap.

The packaging makes the sibling an OPTIONAL dev dependency (available for tests), but
that only makes it importable - it does not stop runtime code from importing it. This
test greps the package source (excluding tests) for the forbidden imports, so CI blocks
the boundary from being crossed rather than leaving it to convention.
"""

from __future__ import annotations

from pathlib import Path
import ast

import skellyforge

_FORBIDDEN = ("skellytracker", "freemocap")


def test_core_tests_and_fixtures_do_not_import_exploratory_tools() -> None:
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for folder in (root / 'skellyforge/tests', root / 'test_support'):
        for path in folder.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                if any(module.split('.')[0] in ('scripts', 'experiments', 'diagnostics') for module in modules):
                    offenders.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not offenders, 'Core tests must use shared fixtures or production code: ' + ', '.join(offenders)


def test_skellyforge_package_never_imports_skellytracker_or_freemocap() -> None:
    package_root = Path(skellyforge.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for name in _FORBIDDEN:
                if f"import {name}" in stripped or f"from {name}" in stripped:
                    offenders.append(
                        f"{path.relative_to(package_root.parent)}:{line_number}: {stripped}"
                    )

    assert not offenders, (
        "skellyforge must not import skellytracker or freemocap at runtime:" + "\n" +
        "\n".join(offenders)
    )
