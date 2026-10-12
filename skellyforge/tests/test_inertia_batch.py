import numpy as np
import pytest
from skellyforge.core.biomechanics.anthropometric_parameters import RadiiOfGyration
from skellyforge.core.biomechanics.segment_inertia import (
    segment_inertia_tensor,
    segment_inertia_tensors,
)


def test_inertia_batch_matches_scalar_and_forward_axis_fallback():
    rng = np.random.default_rng(793)
    proximal = rng.normal(size=(100, 3)) * 1000
    distal = proximal + rng.normal(size=(100, 3)) * 100
    distal[:2] = proximal[:2] + np.array([[0.0, 100.0, 0.0], [0.0, -100.0, 0.0]])
    masses = rng.uniform(1.0, 10.0, 100)
    radii = RadiiOfGyration(transverse=0.3, sagittal=0.2, longitudinal=0.1)
    expected = np.asarray(
        [
            segment_inertia_tensor(mass=float(m), radii=radii, proximal=p, distal=d)
            for m, p, d in zip(masses, proximal, distal, strict=True)
        ]
    )
    actual = segment_inertia_tensors(
        masses=masses,
        radii=np.tile([0.3, 0.2, 0.1], (100, 1)),
        proximal=proximal,
        distal=distal,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-10)


def test_invalid_and_empty_inertia_batches():
    assert segment_inertia_tensors(
        masses=np.empty(0),
        radii=np.empty((0, 3)),
        proximal=np.empty((0, 3)),
        distal=np.empty((0, 3)),
    ).shape == (0, 3, 3)
    with pytest.raises(ValueError, match="zero length"):
        segment_inertia_tensors(
            masses=np.ones(1),
            radii=np.ones((1, 3)),
            proximal=np.zeros((1, 3)),
            distal=np.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="positive"):
        segment_inertia_tensors(
            masses=-np.ones(1),
            radii=np.ones((1, 3)),
            proximal=np.zeros((1, 3)),
            distal=np.ones((1, 3)),
        )
