"""Tag -> colour: how a skeleton is drawn, kept apart from what it is.

A group says what it IS (`tags`); a palette says what that should look like. Keeping the
two apart is what lets a user recolour an entire skeleton by editing one mapping instead of
re-authoring every component file, and it keeps a presentation choice out of a geometry
file that a biomechanist has to read.

Resolution is FIRST MATCH over the group's ordered tags, so specificity is expressed by
ordering rather than by precedence rules the reader has to remember:

    tags ["left_hand", "left"] + palette {left_hand: cyan, left: blue}   -> cyan
    tags ["left_hand", "left"] + palette {left: blue}                    -> blue
    tags []                                                              -> the default

The default is green, and a tag the palette has never heard of gets it too. That is a
DEFAULT, not a fallback: a palette is a partial mapping by design - nobody should have to
enumerate every tag any model might carry - so "no entry" has one obviously-correct answer
and is resolved here. A malformed colour is a different matter and raises at load.

This lives in skellyforge because the things being coloured are skellyforge's: landmarks,
groups, and the `left_`/`right_` sidedness its own loader writes. A skeleton that cannot
say how it is drawn is not standalone, and skellyforge never imports the packages that
would otherwise hold the answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

HEX_COLOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"#[0-9a-fA-F]{6}")
DEFAULT_TAG_COLOR: Final[str] = "#14ff14"
"""What an untagged group, or a tag no palette knows, is drawn in.

Green because that is what FreeMoCap's 2D overlay has always drawn an unclassified
connection in, so an unrecognized tag looks like "not classified yet" rather than like a
deliberate choice.
"""


def raise_unless_hex_color(*, owner: str, color: object) -> str:
    """A `#rrggbb` string, refusing anything else by name.

    A malformed colour reaching a renderer is a silent no-op there, so it is checked where
    the offending entry can still be pointed at.
    """
    if not isinstance(color, str) or not HEX_COLOR_PATTERN.fullmatch(color):
        raise ValueError(
            f"{owner}: colour must be a '#rrggbb' hex string - got {color!r}"
        )
    return color


@dataclass(frozen=True, slots=True, eq=False)
class ColorPalette:
    """A tag -> colour mapping, with one default for everything it does not name.

    Attributes:
        colors_by_tag: the mapping. Partial by design.
        default_color: what an unmatched or untagged thing is drawn in.
    """

    colors_by_tag: Mapping[str, str]
    default_color: str = DEFAULT_TAG_COLOR

    def __post_init__(self) -> None:
        for tag, color in self.colors_by_tag.items():
            if not tag:
                raise ValueError("palette tags must be non-empty strings")
            raise_unless_hex_color(owner=f"palette tag {tag!r}", color=color)
        raise_unless_hex_color(owner="palette default", color=self.default_color)

    def color_for(self, *, tags: Iterable[str]) -> str:
        """The colour of the first tag this palette knows, or its default."""
        for tag in tags:
            color = self.colors_by_tag.get(tag)
            if color is not None:
                return color
        return self.default_color

    def known_tags(self) -> frozenset[str]:
        """Every tag this palette names, for a caller checking its own tags resolve."""
        return frozenset(self.colors_by_tag)

    def with_overrides(self, *, colors_by_tag: Mapping[str, str]) -> ColorPalette:
        """This palette with some tags recoloured — how a user supplies their own scheme.

        Returns a new palette rather than mutating, so a caller cannot recolour another
        caller's skeleton by accident.
        """
        return ColorPalette(
            colors_by_tag={**self.colors_by_tag, **colors_by_tag},
            default_color=self.default_color,
        )

    @classmethod
    def from_yaml(cls, *, path: Path) -> ColorPalette:
        """Load a palette from a YAML file of `tag: '#rrggbb'`, plus an optional default."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(
                f"{path} must parse to a mapping of tag -> colour, got "
                f"{type(document).__name__}"
            )
        entries = dict(document)
        default_color = entries.pop("default", DEFAULT_TAG_COLOR)
        unexpected = sorted(
            tag for tag, color in entries.items() if not isinstance(color, str)
        )
        if unexpected:
            raise ValueError(
                f"{path}: these tags do not map to a colour string - {unexpected}. A "
                "palette is one flat level of `tag: '#rrggbb'`."
            )
        return cls(
            colors_by_tag={str(tag): color for tag, color in entries.items()},
            default_color=default_color,
        )

    @classmethod
    def from_default_yaml(cls) -> ColorPalette:
        """The shipped palette: left/right sides, hands, and calibration-board markers."""
        return cls.from_yaml(
            path=Path(__file__).resolve().parents[3]
            / "definitions"
            / "color_palette.yaml"
        )
