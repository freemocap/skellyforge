"""Loading and validating a named set of coordinate-system conventions from YAML.

The shipped conventions live in definitions/coordinate_systems/coordinate_systems.yaml;
the loader accepts any YAML file in the same shape, so users can define their own
conventions and pass the file in. Loading is fail-loud: an unknown direction, a
non-perpendicular triad, or a declared handedness that contradicts the axes all raise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.core.math.geometry.coordinate_systems.anatomical_direction import (
    parse_anatomical_direction,
)
from skellyforge.core.math.geometry.coordinate_systems.coordinate_system_convention import (
    CoordinateSystemConvention,
)
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness

_AXIS_KEYS: tuple[str, str, str] = ("x_axis", "y_axis", "z_axis")
_CONVENTION_KEYS: frozenset[str] = frozenset({*_AXIS_KEYS, "description", "handedness"})
_HANDEDNESS_BY_LABEL: dict[str, Handedness] = {
    "right": Handedness.RIGHT_HANDED,
    "left": Handedness.LEFT_HANDED,
}


@dataclass(frozen=True, slots=True, eq=False)
class CoordinateSystemRegistry:
    """A named set of coordinate-system conventions and the one the code is authored in.

    Attributes:
        conventions: every convention, keyed by name.
        default_name: the convention the rest of the codebase is authored in.
    """

    conventions: Mapping[str, CoordinateSystemConvention]
    default_name: str

    def __post_init__(self) -> None:
        if not self.conventions:
            raise ValueError("a coordinate system registry needs at least one convention")
        if self.default_name not in self.conventions:
            raise ValueError(
                f"default coordinate system {self.default_name!r} is not one of the "
                f"defined conventions {sorted(self.conventions)}"
            )

    @property
    def default(self) -> CoordinateSystemConvention:
        """The convention the codebase is authored in."""
        return self.conventions[self.default_name]

    def get(self, *, name: str) -> CoordinateSystemConvention:
        """The convention named by name, failing loudly on an unknown name."""
        try:
            return self.conventions[name]
        except KeyError as error:
            raise KeyError(
                f"unknown coordinate system {name!r} - known conventions are "
                f"{sorted(self.conventions)}"
            ) from error

    @classmethod
    def from_yaml(cls, *, path: Path) -> CoordinateSystemRegistry:
        """Load a convention set from a YAML file in the shipped shape."""
        if not path.is_file():
            raise FileNotFoundError(f"coordinate system YAML {path} is not a file")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"{path} must parse to a mapping, got {type(document).__name__}")
        return cls.from_document(document=document, source=str(path))

    @classmethod
    def from_default_yaml(cls) -> CoordinateSystemRegistry:
        """Load the shipped conventions, with Blender as the default."""
        path = (
            Path(__file__).resolve().parents[4]
            / "definitions"
            / "coordinate_systems"
            / "coordinate_systems.yaml"
        )
        return cls.from_yaml(path=path)

    @classmethod
    def from_document(
        cls, *, document: Mapping[str, object], source: str
    ) -> CoordinateSystemRegistry:
        """Build a registry from an already-parsed mapping (the shape from_yaml reads)."""
        default_name = document.get("default")
        if not isinstance(default_name, str) or not default_name:
            raise ValueError(
                f"{source}: needs a non-empty 'default' naming the authored-in convention"
            )

        entries = document.get("conventions")
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError(f"{source}: needs a non-empty 'conventions' mapping")

        conventions = {
            name: _build_convention(name=str(name), entry=entry)
            for name, entry in entries.items()
        }
        return cls(conventions=conventions, default_name=str(default_name))


def _build_convention(*, name: str, entry: object) -> CoordinateSystemConvention:
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"coordinate system {name!r} must be a mapping, got {type(entry).__name__}"
        )
    unexpected_keys = set(entry) - _CONVENTION_KEYS
    if unexpected_keys:
        raise ValueError(
            f"coordinate system {name!r}: unexpected keys {sorted(unexpected_keys)} - "
            f"expected {sorted(_CONVENTION_KEYS)}"
        )

    directions: dict[str, object] = {}
    for axis in _AXIS_KEYS:
        value = entry.get(axis)
        if value is None:
            raise ValueError(
                f"coordinate system {name!r}: missing {axis!r} - each of x_axis, y_axis, "
                f"z_axis is required"
            )
        if not isinstance(value, str):
            raise ValueError(
                f"coordinate system {name!r}: {axis!r} must be a string, got "
                f"{type(value).__name__}"
            )
        directions[axis] = parse_anatomical_direction(label=value)

    convention = CoordinateSystemConvention(
        name=name,
        description=str(entry.get("description", "")),
        x_direction=directions["x_axis"],
        y_direction=directions["y_axis"],
        z_direction=directions["z_axis"],
    )

    declared_handedness = entry.get("handedness")
    if declared_handedness is not None:
        expected = _HANDEDNESS_BY_LABEL.get(str(declared_handedness).lower())
        if expected is None:
            raise ValueError(
                f"coordinate system {name!r}: 'handedness' must be 'right' or 'left', "
                f"got {declared_handedness!r}"
            )
        if expected is not convention.handedness:
            raise ValueError(
                f"coordinate system {name!r}: declared handedness "
                f"{str(declared_handedness)!r} contradicts its axes "
                f"({directions['x_axis'].value}, {directions['y_axis'].value}, "
                f"{directions['z_axis'].value})"
            )
    return convention
