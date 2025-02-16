from indicators.base.core import CoreIndicators
import numpy as np


class xDecay(CoreIndicators):
    """
    Exponential smoothing for financial time series.

    Exponential smoothing is a technique used for smoothing time series data by giving more 
    weight to recent data points and less weight to older data points. It is useful for 
    identifying trends and seasonality in the data.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - dimension (str): The column name representing the dimension in the timeseries.
    - alpha (float): The smoothing factor, which determines the weight given to recent 
    observations. Should be between 0 and 1.

    Returns:
    - numpy.ndarray: An array containing the smoothed values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, dimension, alpha):
        dimension     = self._extractor.extract_dimension(timeseries, dimension)
        index         = np.argmax(~np.isnan(dimension))
        xdecay        = np.full(dimension.shape[0], np.nan)
        xdecay[index] = dimension[index]
        index        += 1
        for i in range(index, dimension.shape[0]):
            xdecay[i] = (1 - alpha) * xdecay[i-1] + alpha * dimension[i]
        return xdecay
