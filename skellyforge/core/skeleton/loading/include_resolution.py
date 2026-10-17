"""Stage 1 of loading: resolve $include directives.

A mapping whose only key is $include is replaced by the parsed contents of the file it
names, resolved relative to the INCLUDING file's directory, so an include means the same
thing no matter who included it. Includes are cycle-guarded.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import yaml

INCLUDE_KEY: Final[str] = "$include"


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: includes
# ═══════════════════════════════════════════════════════════════════════


def resolve_includes(*, node: object, base_directory: Path, include_stack: tuple[Path, ...]) -> object:
    """Replace every `{$include: path}` mapping with the parsed contents of that file.

    Args:
        node: the parsed YAML node to walk.
        base_directory: directory that relative include paths are resolved against - the
            directory of the file this node came from, so an include means the same thing
            no matter who included it.
        include_stack: files currently being included, innermost last, for cycle detection.

    Returns:
        The node with every include replaced by the document it names.

    Raises:
        ValueError: an include mapping carries other keys, or the includes form a cycle.
        FileNotFoundError: an included path does not exist.
    """
    if isinstance(node, Mapping):
        if INCLUDE_KEY in node:
            if len(node) != 1:
                raise ValueError(
                    f"an `{INCLUDE_KEY}` mapping means 'paste this file here' and so may "
                    f"carry no other keys - got {sorted(node)}"
                )
            return _load_included_file(
                relative_path=str(node[INCLUDE_KEY]),
                base_directory=base_directory,
                include_stack=include_stack,
            )
        return {
            key: resolve_includes(
                node=value, base_directory=base_directory, include_stack=include_stack
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            resolve_includes(
                node=item, base_directory=base_directory, include_stack=include_stack
            )
            for item in node
        ]
    return node


def _load_included_file(
    *, relative_path: str, base_directory: Path, include_stack: tuple[Path, ...]
) -> object:
    """Parse one included file and recursively resolve the includes inside it."""
    included_path = (base_directory / relative_path).resolve()
    if included_path in include_stack:
        cycle = " -> ".join(path.name for path in (*include_stack, included_path))
        raise ValueError(f"`{INCLUDE_KEY}` cycle: {cycle}")
    if not included_path.is_file():
        raise FileNotFoundError(
            f"`{INCLUDE_KEY}: {relative_path}` (from {base_directory}) does not name a "
            f"file - looked at {included_path}"
        )
    return resolve_includes(
        node=yaml.safe_load(included_path.read_text(encoding="utf-8")),
        base_directory=included_path.parent,
        include_stack=(*include_stack, included_path),
    )
