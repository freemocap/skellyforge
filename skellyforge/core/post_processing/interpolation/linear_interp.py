
import numpy as np
import pandas as pd
from skellyforge.post_processing.interpolation.interpolation_config import InterpolationConfig

def linear_interpolate(data:np.ndarray, config:InterpolationConfig) -> np.ndarray:    
    df = pd.DataFrame(data)
    df2 = df.interpolate(method = 'linear',axis = 0) #use pandas interpolation methods to fill in missing data
    return np.array(df2)

