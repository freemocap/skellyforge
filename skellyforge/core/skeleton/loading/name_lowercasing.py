"""Stage 2 of loading: lowercase the names in a component document.

Keys are lowercased unconditionally; the values listed in NAME_VALUED_KEYS (names and
keywords) are too. Prose, numbers and paths are left as written, so an unrecognized field
fails safe by staying untouched rather than being silently mangled.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

NAME_VALUED_KEYS: Final[frozenset[str]] = frozenset(
    # `landmark_names` and `pairs` are the grouping sections' member lists. They are named
    # distinctly from the `landmarks:` SECTION on purpose: that section's value is a mapping
    # of entries, which this would pass through untouched, silently leaving every landmark
    # in the file un-lowercased.
    {"name", "aliases", "reference_frame", "origin", "landmark", "type", "landmark_names", "pairs"}
)


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: lowercasing
# ═══════════════════════════════════════════════════════════════════════


def lowercase_names(*, node: object) -> object:
    """Lowercase every KEY, and those values that are names.

    The SCREAMING_SNAKE in the YAML is there to make the document scannable; the canonical
    runtime name is lowercase. Every landmark and segment name is a key, so keys are
    lowercased unconditionally. Values are a different matter: `NAME_VALUED_KEYS` lists the
    keys whose values are names or keywords, and only those are touched.

    An earlier version lowercased every string in the tree and exempted `definition`. That
    is the same thing for today's document, and the wrong rule: it made "is this text a
    name?" depend on a blocklist that any new field would have to remember to join, and it
    silently mangled anything that was neither - a citation, a units string, a path. An
    allowlist fails the safe way round, by leaving an unrecognized value alone.
    """
    if isinstance(node, Mapping):
        lowercased: dict[str, object] = {}
        for key, value in node.items():
            lowercased_key = str(key).lower()
            lowercased[lowercased_key] = (
                _lowercased_names_in(value=value)
                if lowercased_key in NAME_VALUED_KEYS
                else lowercase_names(node=value)
            )
        return lowercased
    if isinstance(node, list):
        return [lowercase_names(node=item) for item in node]
    return node


def _lowercased_names_in(*, value: object) -> object:
    """One name, or a list of them, lowercased. Anything else is passed through."""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return [_lowercased_names_in(value=item) for item in value]
    return value


def _as_list(*, name: str, field_name: str, value: object) -> list[object]:
    """A YAML sequence field as a list, refusing a bare scalar written by mistake."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name!r}: `{field_name}` must be a list - got {type(value).__name__} "
            f"({value!r})"
        )
    return list(value)
