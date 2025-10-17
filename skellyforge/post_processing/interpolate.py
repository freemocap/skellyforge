from pydantic import BaseModel
from skellyforge.data_models.data_3d import Trajectory3d

class InterpolationConfig(BaseModel):
    method: str = "linear"
    order: int = 3

def interpolate_trajectory(trajectory:Trajectory3d, 
                           config: InterpolationConfig):
    data = trajectory.triangulated_data