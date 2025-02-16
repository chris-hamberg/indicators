from indicators.base.core import CoreIndicators
from statsmodels.tsa.stattools import acf
import numpy as np


class Seismology(CoreIndicators):
    """
    Computes a matrix combining autocorrelation coefficients with the original 
    time series values.

    The matrix provides a representation of the relationship between past values 
    in the time series, which could be valuable in detecting patterns or trends 
    that are relevant for forecasting or trading strategies.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - n (int): The number of periods to consider in the computation.
    - dimension (str): The column name or key representing the dimension of 
    interest in the time series.

    Returns:
    - numpy.ndarray: A matrix combining autocorrelation coefficients with the 
    original time series values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        matrix, pad = [], np.full((n, n), np.nan)
        for i in range(n, dimension.shape[0]):
            subset   = dimension[i-n+1:i+1]
            autocore = acf(subset, nlags=n)[::-1]
            autocore = np.hstack((autocore[:-1], (np.nan,)))
            combination = (autocore * subset)
            matrix.append(combination)
        matrix = np.array(matrix)
        matrix = np.concatenate((pad, matrix))
        return matrix
