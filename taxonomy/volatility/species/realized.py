from indicators.base.core import CoreIndicators
import numpy as np


class RealizedVolatility(CoreIndicators):
    """
    Computes the realized volatility of asset prices over a given period.

    The algorithm performs the following steps:

        1. Extracts the specified dimension (e.g., price) from the timeseries.
        2. Calculates the log returns of consecutive trade prices.
                log_return[t] = log(price[t] / price[t-1])
        3. Handles extreme log return values (infinite values) by replacing them 
           with NaN.
        4. Squares the log returns to obtain variance contributions.
        5. Sums the squared log returns over the desired period using the 
           summation method.
        6. If the summed squared values are negative, they are imputed to zero 
           to avoid errors when calculating the square root.
        7. Takes the square root of the sum to obtain the realized volatility.

    This measure reflects the magnitude of price fluctuations over time,
    providing insight into the asset's volatility based on observed trade 
    prices.

    Parameters:
    - timeseries: Array-like object (e.g., list, numpy array) containing the 
      price data for the asset.
    - n: Integer representing the lookback period (e.g., number of minutes).
    - dimension: The specific dimension (e.g., 'Close') to calculate volatility 
      on from the timeseries.

    Returns:
    - Realized volatility as a numpy array with values representing the 
      volatility for each period.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):

        dimension = self._extractor.extract_dimension(timeseries, dimension)
        LogR      = self._LogR(dimension)
        imputed   = np.where(np.abs(LogR) == np.inf, np.nan, LogR)
        squared   = imputed ** 2
        summation = self._summation(squared, n)
        imputed   = np.where(summation < 0, 0, summation)
        realized  = np.sqrt(imputed)

        return realized
