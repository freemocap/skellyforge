"""Tests for the anthropometric-parameter validation code.

These exercise the code's own validation rules with synthetic documents. The
shipped de Leva table is data, not behavior, so its values are never asserted here.
"""

from __future__ import annotations

import pytest

from skellyforge.core.biomechanics.anthropometric_parameters import (
    AnthropometricParameters,
    AnthropometricSegment,
    RadiiOfGyration,
)


def _parameters() -> AnthropometricParameters:
    return AnthropometricParameters.from_default_yaml()


def test_get_fails_on_an_unknown_segment() -> None:
    with pytest.raises(KeyError):
        _parameters().get(name="spleen")


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
