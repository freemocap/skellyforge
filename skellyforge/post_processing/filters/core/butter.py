from scipy import signal
import numpy as np
from skellyforge.post_processing.filters.filter_config import FilterConfig

def butter_low_pass(data:np.ndarray, 
                    cutoff_freq:float, 
                    sampling_rate: float, 
                    order:int) -> np.ndarray:
    """ Run a low pass butterworth filter on a single column of data"""
    nyquist_freq = 0.5*sampling_rate
    normal_cutoff = cutoff_freq/nyquist_freq

    b,a = signal.butter(order, normal_cutoff, btype = 'low', analog=False)
    return signal.filtfilt(b,a,data)

def butter_from_config(
        data:np.ndarray,
        config: FilterConfig
):
    return butter_low_pass(
        data,
        config.cutoff,
        config.sampling_rate,
        config.order
    )