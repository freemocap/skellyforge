from pydantic import BaseModel
from enum import Enum

class FilterMethod(str, Enum):
    butter_low_pass = "butter_low_pass"

class FilterConfig(BaseModel):
    method:FilterMethod = FilterMethod.butter_low_pass
    cutoff:float = 6.0 #in Hz
    sampling_rate:float = 30.0 #in Hz
    order:int = 4
