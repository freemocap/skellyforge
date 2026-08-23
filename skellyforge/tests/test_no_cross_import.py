"""Enforce the runtime boundary: skellyforge must never import skellytracker or freemocap.

The packaging makes the sibling an OPTIONAL dev dependency (available for tests), but
that only makes it importable - it does not stop runtime code from importing it. This
test greps the package source (excluding tests) for the forbidden imports, so CI blocks
the boundary from being crossed rather than leaving it to convention.
"""

from __future__ import annotations

from pathlib import Path

import skellyforge

_FORBIDDEN = ("skellytracker", "freemocap")


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
