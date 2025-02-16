from indicators.base.core import CoreIndicators
import numpy as np
import pywt


class Wavelet(CoreIndicators):
    """
    Wavelet smoothing for financial time series.

    Wavelet smoothing is a technique that uses wavelet transform to decompose a time series 
    into different frequency components. This can help in removing noise and extracting 
    underlying trends.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - dimension (str): The column name representing the dimension in the timeseries.
    - wavelet (str): The wavelet function to use for decomposition. Common choices include 
    'haar', 'db2', 'sym5', etc.
    - level (int): The level of decomposition. Higher levels capture more details but may 
    also include more noise.

    Returns:
    - numpy.ndarray: An array containing the smoothed values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, dimension, wavelet, level):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        coefficients   = pywt.wavedec(dimension, wavelet, level=level)
        decomposition  = [coefficients[0]]
        decomposition += [np.zeros_like(detail) for detail in coefficients[1:]]
        return pywt.waverec(decomposition, wavelet)[-dimension.shape[0]:]
