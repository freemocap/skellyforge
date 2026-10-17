"""Tag -> colour resolution.

The palette exists so a user can recolour a whole skeleton by editing one mapping. These
tests pin the two properties that make that true: resolution is first-match over ordered tags
(so specificity lives in the model's ordering, not in precedence rules here), and anything
unnamed resolves to one default rather than to nothing. The shipped colour scheme itself is
data, so it is not asserted here.
"""

from __future__ import annotations

import pytest

from skellyforge.core.skeleton.components.color_palette import (
    DEFAULT_TAG_COLOR,
    ColorPalette,
)


# ── resolution ─────────────────────────────────────────────────────────────


def test_the_first_known_tag_wins() -> None:
    """Specificity is the model's ordering, so the palette needs no precedence rules."""
    palette = ColorPalette(colors_by_tag={"left_hand": "#22dddd", "left": "#4488ff"})
    assert palette.color_for(tags=("left_hand", "left")) == "#22dddd"
    assert palette.color_for(tags=("left", "left_hand")) == "#4488ff"


def test_an_unknown_leading_tag_falls_through_to_the_next() -> None:
    """A palette is partial by design — nobody enumerates every tag a model might carry."""
    palette = ColorPalette(colors_by_tag={"face": "#ffd166"})
    assert palette.color_for(tags=("eye", "face")) == "#ffd166"


def test_no_tags_and_no_matches_both_get_the_default() -> None:
    palette = ColorPalette(colors_by_tag={"face": "#ffd166"})
    assert palette.color_for(tags=()) == DEFAULT_TAG_COLOR
    assert palette.color_for(tags=("nothing_it_knows",)) == DEFAULT_TAG_COLOR


def test_a_malformed_colour_is_refused_where_it_is_named() -> None:
    """A bad colour is a silent no-op in a renderer, so it fails at load instead."""
    with pytest.raises(ValueError, match="palette tag 'face'"):
        ColorPalette(colors_by_tag={"face": "reddish"})
    with pytest.raises(ValueError, match="palette default"):
        ColorPalette(colors_by_tag={}, default_color="#fff")


def test_overrides_produce_a_new_palette_without_touching_the_old_one() -> None:
    """A user's scheme must not recolour somebody else's skeleton."""
    base = ColorPalette(colors_by_tag={"left": "#4488ff"})
    recoloured = base.with_overrides(colors_by_tag={"left": "#000000"})

    assert recoloured.color_for(tags=("left",)) == "#000000"
    assert base.color_for(tags=("left",)) == "#4488ff"
