"""A durable scale fit must reject invalid numeric values and segment bindings."""

from dataclasses import replace

import pytest

from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit


def fit_fixture() -> ModelScaleFit:
    return ModelScaleFit(
        fitted_scale=100.0,
        segment_scales={"segment": 100.0},
        segment_lengths={"segment": 10.0},
        measured_segment_names=frozenset({"segment"}),
        voting_segment_names=frozenset({"segment"}),
    )


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0.0, -1.0])
def test_fit_rejects_invalid_scale(value: float) -> None:
    with pytest.raises(ValueError):
        replace(fit_fixture(), fitted_scale=value)
    with pytest.raises(ValueError):
        replace(fit_fixture(), segment_scales={"segment": value})


def test_fit_rejects_inconsistent_segments() -> None:
    with pytest.raises(ValueError, match="same nonempty segment set"):
        replace(fit_fixture(), segment_lengths={"another": 10.0})
    with pytest.raises(ValueError, match="Measured segments"):
        replace(fit_fixture(), measured_segment_names=frozenset({"another"}))
    with pytest.raises(ValueError, match="Voting segments"):
        replace(fit_fixture(), voting_segment_names=frozenset({"another"}))


def test_fit_uses_value_equality() -> None:
    assert fit_fixture() == fit_fixture()
