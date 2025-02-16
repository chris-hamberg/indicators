from indicators.base.core import CoreIndicators
import numpy as np


class Sinusoidal(CoreIndicators):
    """
    Generate a sinusoidal signal based on the timestamps of a financial time series.

    The Sinusoidal indicator generates a sinusoidal signal based on the timestamps of 
    a financial time series. It can be used to model cyclic patterns in the data.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - cycle (int): The duration of one complete cycle of the sinusoidal signal.
    - granularity (int): The number of timestamps per unit cycle.
    - amplitude (float): The amplitude of the sinusoidal signal.
    - offset (float): The offset or phase shift of the sinusoidal signal.

    Returns:
    - numpy.ndarray: An array containing the generated sinusoidal signal.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, cycle, granularity, amplitude, offset):
        timestamps  = self._extractor.extract_matrix(timeseries)[:,0]
        periodicity = cycle / granularity
        timestamps  = timestamps.astype(int)
        sinusoidal  = amplitude * (2 * np.pi * timestamps * periodicity) + offset
        return sinusoidal
