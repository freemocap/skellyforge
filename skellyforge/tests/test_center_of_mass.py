"""Tests for the center-of-mass validation code.

These exercise the code's own validation rules with synthetic documents. The
shipped configuration files are data, not behavior, so their contents are never
asserted here.
"""

from __future__ import annotations

import pytest

from skellyforge.core.biomechanics.center_of_mass import CenterOfMassDefinitions


def test_rejects_weights_that_do_not_sum_to_one() -> None:
    document = {
        "segments": {
            "head_neck": {
                "proximal": "cervicothoracic_junction",
                "distal": "head_vertex",
                "landmarks": [{"landmark": "head_center", "weight": 0.9}],
            }
        }
    }
    with pytest.raises(ValueError):
        CenterOfMassDefinitions.from_document(document=document, source="test")


def test_rejects_mixing_bare_and_weighted_landmarks() -> None:
    document = {
        "segments": {
            "head_neck": {
                "proximal": "cervicothoracic_junction",
                "distal": "head_vertex",
                "landmarks": ["head_center", {"landmark": "chin", "weight": 0.5}],
            }
        }
    }
    with pytest.raises(ValueError):
        CenterOfMassDefinitions.from_document(document=document, source="test")
