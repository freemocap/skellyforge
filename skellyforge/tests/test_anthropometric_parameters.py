"""Tests for the de Leva (1996) anthropometric parameter loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from skellyforge.core.biomechanics.anthropometric_parameters import (
    AnthropometricParameters,
    AnthropometricSegment,
    RadiiOfGyration,
)

EXPECTED_UNIQUE_SEGMENTS: int = 10
EXPECTED_BILATERAL_SEGMENTS: frozenset[str] = frozenset(
    {"upper_arm", "forearm", "hand", "thigh", "shank", "foot"}
)


def _parameters() -> AnthropometricParameters:
    return AnthropometricParameters.from_default_yaml()


def test_mass_fractions_counting_bilateral_segments_twice_sum_to_one() -> None:
    parameters = _parameters()
    total = sum(
        segment.mass_fraction * segment.side_count
        for segment in parameters.segments.values()
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_there_are_ten_unique_anatomical_segments() -> None:
    assert len(_parameters().segments) == EXPECTED_UNIQUE_SEGMENTS


def test_bilateral_segments_are_flagged() -> None:
    parameters = _parameters()
    for name, segment in parameters.segments.items():
        assert segment.bilateral == (name in EXPECTED_BILATERAL_SEGMENTS)


def test_get_returns_a_segment_and_fails_on_unknown() -> None:
    parameters = _parameters()
    assert parameters.get(name="thigh").mass_fraction == pytest.approx(0.1416)
    with pytest.raises(KeyError):
        parameters.get(name="spleen")


def test_rejects_a_mass_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        AnthropometricSegment(
            name="bad",
            mass_fraction=1.5,
            center_of_mass_fraction=0.5,
            radii_of_gyration=RadiiOfGyration(
                sagittal=0.3, transverse=0.3, longitudinal=0.3
            ),
        )


def test_rejects_a_radius_of_gyration_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        RadiiOfGyration(sagittal=1.2, transverse=0.3, longitudinal=0.3)


def test_rejects_a_table_whose_masses_do_not_sum_to_one() -> None:
    document = {
        "segments": {
            "head_neck": {
                "mass_fraction": 0.5,
                "center_of_mass_fraction": 0.5,
                "radii_of_gyration": {
                    "sagittal": 0.3,
                    "transverse": 0.3,
                    "longitudinal": 0.3,
                },
            }
        }
    }
    with pytest.raises(ValueError):
        AnthropometricParameters.from_document(document=document, source="test")
