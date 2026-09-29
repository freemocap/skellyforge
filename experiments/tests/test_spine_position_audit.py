"""Numerical convergence cannot hide a large axial landmark displacement."""
import numpy as np
from experiments.generators.solver_spine_positions import audit, LANDMARKS, ACCEPTANCE_DISTANCE_MM


def test_tail_escape_fails_acceptance_even_when_solver_converged():
    frames=[dict(bodies=[dict(fitted=[[0.,0.,0.]]*len(LANDMARKS))],diagnostics={}) for _ in range(3)]
    frames[-1]['bodies'][0]['fitted'][1]=[0.,0.,ACCEPTANCE_DISTANCE_MM+1]
    method=dict(frames=frames,summary={},converged=True)
    candidate=dict(bodies=[dict(landmark_names=list(LANDMARKS))],runs=[dict(methods=dict(full_body=method))])
    records=[dict(number=i,points={name:np.zeros(3) for name in LANDMARKS}) for i in range(3)]
    result=audit(candidate,records)
    assert not result['passed']
    assert result['landmarks']['chest_center']['outside_limit_frames']==[2]
    assert result['landmarks']['chest_center']['worst_frame']==2
    assert method['converged']  # preserve numerical status, report geometry separately
