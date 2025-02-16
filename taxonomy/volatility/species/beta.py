from indicators.base.core import CoreIndicators
import numpy as np


class Beta(CoreIndicators):
    """Calculates the beta coefficient between two time series.

    Beta measures the volatility of a stock relative to the market or another asset.
    A beta of 1 indicates the stock has the same volatility as the market,
    while a beta greater than 1 indicates higher volatility and less than 1 indicates lower 
    volatility.

    Parameters:
    - timeseriesA (pd.DataFrame or Stock): The first input financial time series.
    - timeseriesB (pd.DataFrame or Stock): The second input financial time series.
    - n (int): The lookback period for calculating beta.
    - dimensionA (str): The dimension name in timeseriesA to be used in the calculation.
    - dimensionB (str): The dimension name in timeseriesB to be used in the calculation.
    - LogR (bool): If True, calculate beta based on log returns. Default is True.

    Returns:
    - beta (float): The beta coefficient representing the volatility relationship between the 
    two time series.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseriesA, timeseriesB, n, dimensionA, dimensionB, LogR):
        if not dimensionB: dimensionB = dimensionA
        A = self._extractor.extract_dimension(timeseriesA, dimensionA)
        B = self._extractor.extract_dimension(timeseriesB, dimensionB)
        if LogR: A, B = self._LogR(A), self._LogR(B)
        A, B = A[-n:], B[-n:]
        covar = np.cov(A, B)[0, 1]
        var   = np.nanvar(B)
        beta  = covar / var
        return beta
