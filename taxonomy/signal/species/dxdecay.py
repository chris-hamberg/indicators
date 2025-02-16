from indicators.base.core import CoreIndicators
import numpy as np


class dxDecay(CoreIndicators):
    """
    Double Exponential Smoothing for financial time series.

    Double Exponential Smoothing is a method used for time series forecasting. It 
    is an extension of the popular Exponential Smoothing method, adding a 
    trend-smoothing component to the model.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - dimension (str): The column name representing the dimension in the timeseries.
    - alpha (float): Smoothing factor for the level component, between 0 and 1.
    - beta (float): Smoothing factor for the trend component, between 0 and 1.

    Returns:
    - numpy.ndarray: An array containing the computed values of the Double Exponential
    Smoothing for the given timeseries.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, dimension, alpha, beta):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        index = np.argmax(~np.isnan(dimension))
        level = np.zeros(dimension.shape[0])
        trend = np.zeros(dimension.shape[0])
        level[index] = dimension[index]
        trend[index] = dimension[index+1] - dimension[index]
        index += 1
        for i in range(index, dimension.shape[0]):
            level[i] = alpha * dimension[i] + (1-alpha) * (level[i-1] + trend[i-1])
            trend[i] = beta * (level[i] - level[i-1]) + (1-beta) * trend[i-1]
        return np.array(level + trend)
