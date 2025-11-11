from pydantic import BaseModel, ConfigDict
import numpy as np


class Trajectory2d(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    start_frame: int
    end_frame: int
    points_2d: np.ndarray