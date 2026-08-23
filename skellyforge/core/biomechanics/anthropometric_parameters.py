"""Anthropometric segment parameters (de Leva 1996): mass and radii of gyration.

This is the pure-data layer of the biomechanics module. It knows anatomical segment names
and fractions; it knows nothing about the skeleton's landmarks or poses. The
skeleton-facing pieces - which landmarks define each segment's center of mass, and how the
61 skeleton segments map onto these anatomical segments - live in center_of_mass.py and
segment_mapping.py respectively.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.core.biomechanics import FRACTION_SUM_TOLERANCE


@dataclass(frozen=True, slots=True)
class RadiiOfGyration:
    """Radii of gyration about a segment's principal axes, as fractions of segment length.

    Attributes:
        sagittal: about the anterior-posterior axis.
        transverse: about the medio-lateral axis.
        longitudinal: about the bone's long axis (proximal -> distal).
    """

    sagittal: float
    transverse: float
    longitudinal: float

    def __post_init__(self) -> None:
        for axis, value in (
            ("sagittal", self.sagittal),
            ("transverse", self.transverse),
            ("longitudinal", self.longitudinal),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"radius of gyration about the {axis} axis must be in (0, 1), got {value}"
                )


@dataclass(frozen=True, slots=True, eq=False)
class AnthropometricSegment:
    """One anatomical segment's inertial parameters.

    Attributes:
        name: snake_case anatomical segment name.
        mass_fraction: fraction of TOTAL body mass this segment carries (per side when
            bilateral).
        center_of_mass_fraction: the COM's distance from the proximal end, as a fraction
            of segment length. Reference only; the authoritative 3D COM is authored in
            center_of_mass.yaml.
        radii_of_gyration: radii of gyration about the principal axes, fractions of length.
        bilateral: whether the segment exists as a left/right pair (limbs).
    """

    name: str
    mass_fraction: float
    center_of_mass_fraction: float
    radii_of_gyration: RadiiOfGyration
    bilateral: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.mass_fraction < 1.0:
            raise ValueError(
                f"segment {self.name!r}: mass_fraction must be in (0, 1), got "
                f"{self.mass_fraction}"
            )
        if not 0.0 < self.center_of_mass_fraction < 1.0:
            raise ValueError(
                f"segment {self.name!r}: center_of_mass_fraction must be in (0, 1), got "
                f"{self.center_of_mass_fraction}"
            )

    @property
    def side_count(self) -> int:
        """1 for a midline segment, 2 for a bilateral (left/right) pair."""
        return 2 if self.bilateral else 1


@dataclass(frozen=True, slots=True, eq=False)
class AnthropometricParameters:
    """Every anatomical segment's inertial parameters, keyed by anatomical name."""

    segments: Mapping[str, AnthropometricSegment]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("anthropometric parameters need at least one segment")
        for name, segment in self.segments.items():
            if name != segment.name:
                raise ValueError(
                    f"anthropometric segments must be keyed by their own name - got key "
                    f"{name!r} for segment {segment.name!r}"
                )
        total = sum(
            segment.mass_fraction * segment.side_count
            for segment in self.segments.values()
        )
        if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
            raise ValueError(
                f"mass fractions must sum to 1.0 (counting bilateral segments twice) - "
                f"got {total}"
            )

    def get(self, *, name: str) -> AnthropometricSegment:
        """The anatomical segment named by name, failing loudly on an unknown name."""
        try:
            return self.segments[name]
        except KeyError as error:
            raise KeyError(
                f"unknown anatomical segment {name!r} - known segments are "
                f"{sorted(self.segments)}"
            ) from error

    @classmethod
    def from_yaml(cls, *, path: Path) -> AnthropometricParameters:
        """Load anthropometric parameters from a YAML file in the shipped shape."""
        if not path.is_file():
            raise FileNotFoundError(f"anthropometric parameters YAML {path} is not a file")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(
                f"{path} must parse to a mapping, got {type(document).__name__}"
            )
        return cls.from_document(document=document, source=str(path))

    @classmethod
    def from_default_yaml(cls) -> AnthropometricParameters:
        """Load the shipped de Leva (1996) parameters for the standard human."""
        path = (
            Path(__file__).resolve().parents[2]
            / "definitions"
            / "human_skeleton"
            / "anthropometric_parameters.yaml"
        )
        return cls.from_yaml(path=path)

    @classmethod
    def from_document(
        cls, *, document: Mapping[str, object], source: str
    ) -> AnthropometricParameters:
        """Build parameters from an already-parsed mapping (the shape from_yaml reads)."""
        entries = document.get("segments")
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError(f"{source}: needs a non-empty 'segments' mapping")

        segments: dict[str, AnthropometricSegment] = {}
        for name, entry in entries.items():
            segments[str(name)] = _build_segment(name=str(name), entry=entry)
        return cls(segments=segments)


_RADII_KEYS: frozenset[str] = frozenset({"sagittal", "transverse", "longitudinal"})
_SEGMENT_KEYS: frozenset[str] = frozenset(
    {"mass_fraction", "center_of_mass_fraction", "radii_of_gyration", "bilateral"}
)


def _build_segment(*, name: str, entry: object) -> AnthropometricSegment:
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"anthropometric segment {name!r} must be a mapping, got "
            f"{type(entry).__name__}"
        )
    unexpected_keys = set(entry) - _SEGMENT_KEYS
    if unexpected_keys:
        raise ValueError(
            f"anthropometric segment {name!r}: unexpected keys {sorted(unexpected_keys)} - "
            f"expected {sorted(_SEGMENT_KEYS)}"
        )

    mass_fraction = entry.get("mass_fraction")
    if not isinstance(mass_fraction, (int, float)):
        raise ValueError(
            f"anthropometric segment {name!r}: 'mass_fraction' must be a number, got "
            f"{type(mass_fraction).__name__}"
        )

    center_of_mass_fraction = entry.get("center_of_mass_fraction")
    if not isinstance(center_of_mass_fraction, (int, float)):
        raise ValueError(
            f"anthropometric segment {name!r}: 'center_of_mass_fraction' must be a "
            f"number, got {type(center_of_mass_fraction).__name__}"
        )

    radii_entry = entry.get("radii_of_gyration")
    if not isinstance(radii_entry, Mapping):
        raise ValueError(
            f"anthropometric segment {name!r}: 'radii_of_gyration' must be a mapping"
        )
    missing = _RADII_KEYS - set(radii_entry)
    if missing:
        raise ValueError(
            f"anthropometric segment {name!r}: missing radii {sorted(missing)}"
        )
    unexpected_radii = set(radii_entry) - _RADII_KEYS
    if unexpected_radii:
        raise ValueError(
            f"anthropometric segment {name!r}: unexpected radii {sorted(unexpected_radii)}"
        )

    bilateral = entry.get("bilateral", False)
    if not isinstance(bilateral, bool):
        raise ValueError(
            f"anthropometric segment {name!r}: 'bilateral' must be a boolean, got "
            f"{type(bilateral).__name__}"
        )

    return AnthropometricSegment(
        name=name,
        mass_fraction=float(mass_fraction),
        center_of_mass_fraction=float(center_of_mass_fraction),
        radii_of_gyration=RadiiOfGyration(
            sagittal=float(radii_entry["sagittal"]),
            transverse=float(radii_entry["transverse"]),
            longitudinal=float(radii_entry["longitudinal"]),
        ),
        bilateral=bilateral,
    )
