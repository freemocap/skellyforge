from pydantic import BaseModel
import numpy as np


class Trajectory2d(BaseModel):
    start_frame: int
    end_frame: int
    points_2d: np.ndarray