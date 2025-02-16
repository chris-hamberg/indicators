from indicators.base.core import CoreIndicators
import numpy as np


class txDecay(CoreIndicators):
    """
    Triple Exponential Smoothing for time series forecasting.

    Triple Exponential Smoothing, also known as Holt-Winters method, is a popular 
    technique for forecasting time series data with trends and seasonalities. It 
    uses three smoothing parameters (alpha, beta, gamma) to model the level, trend, 
    and seasonality of the time series.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - dimension (str): The column name representing the dimension in the timeseries.
    - alpha (float): The smoothing factor for the level component. Should be between 0 and 1.
    - beta (float): The smoothing factor for the trend component. Should be between 0 and 1.
    - gamma (float): The smoothing factor for the seasonal component. Should be between 0 and 1.
    - seasonality (int): The length of the seasonal cycle in the time series.

    Returns:
    - numpy.ndarray: An array containing the smoothed values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, dimension, alpha, beta, gamma, seasonality):
        dimension = self._extractor.extract_dimension(timeseries, dimension)

        x        = np.argmax(~np.isnan(dimension))
        level    = np.zeros(dimension.shape[0])
        trend    = np.zeros(dimension.shape[0])
        season   = np.zeros(dimension.shape[0])
        level[x] = dimension[x]
        trend[x] = dimension[x+1] - dimension[x]
        season[x:x+seasonality] = dimension[x:x+seasonality] - level[x]
        x += 1 

        for i in range(x, dimension.shape[0]):
            
            period = season[i % seasonality]

            level[i] = alpha * (dimension[i] - period) + (1-alpha) * level[i-1]

            trend[i] = beta * (level[i] - level[i-1]) + (1-beta) * trend[i-1]

            gterm = dimension[i] - level[i]
            season[i % seasonality] = gamma * gterm + (1 - gamma) * period

        return np.array(level + trend + season)
