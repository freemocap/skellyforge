from pydantic import BaseModel
from enum import Enum

class InterpolationMethod(str, Enum):
    linear = "linear"

class InterpolationConfig(BaseModel):
    method: InterpolationMethod = InterpolationMethod.linear
