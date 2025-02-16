from indicators.base.core import CoreIndicators
import numpy as np


class QuarkSpectrum(CoreIndicators):
    """
    Quark Spectrum indicator.

    Quark Spectrum is a technical indicator that measures the peak-to-peak (PTP) amplitude of 
    a rolling window of a given dimension in a timeseries. It can help identify periods of 
    increased volatility or amplitude in the data.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which should include 
    columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the rolling window.
    - dimension (str): The column name representing the dimension in the timeseries. Typically, 
    'close' is used for this parameter.

    Returns:
    numpy.ndarray: An array containing the calculated Quark Spectrum values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension   = self._extractor.extract_dimension(timeseries, dimension)
        window, ptp = self._windows.rolling(dimension, n)
        ptp[n-1:]   = np.ptp(window, axis=1)
        return ptp
