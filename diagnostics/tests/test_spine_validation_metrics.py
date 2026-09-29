import numpy as np
import pytest
from scripts.solver_spine_validation import change_metrics, angular_rates


@pytest.mark.parametrize('fps',[6,30,120])
def test_constant_speed_metrics_use_seconds_and_ignore_quaternion_sign(fps):
    times=np.arange(fps+1)/fps
    linear=change_metrics(40*times,times)
    assert linear['rate_p95_mm_s']==pytest.approx(40)
    assert linear['step_max_mm']==pytest.approx(40/fps)
    half_angle=np.deg2rad(60*times)/2
    q=np.column_stack((np.cos(half_angle),np.zeros_like(times),np.zeros_like(times),np.sin(half_angle)))
    q[::2]*=-1
    np.testing.assert_allclose(angular_rates(q,times),60,atol=1e-7)


def test_nonfinite_motion_cannot_pass_metrics():
    with pytest.raises(ValueError,match='Finite values'):
        change_metrics([0,np.nan,1],[0,1,2])
