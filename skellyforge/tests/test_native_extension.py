"""Exercise the installed compiled boundary, not skeleton fitting."""

import pytest
from skellyforge import _native


def test_ceres_autodiff_solve():
    result = _native.solve_scalar(initial=-12.0, target=7.0)
    assert _native.ceres_version == "2.2.0"
    assert result.converged
    assert result.value == pytest.approx(7.0, abs=1e-6)
    assert result.final_cost < 1e-12


def test_native_error_reaches_python():
    with pytest.raises(ValueError, match="finite"):
        _native.solve_scalar(initial=float("nan"), target=7.0)
