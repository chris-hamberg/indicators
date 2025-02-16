from indicators.base.core import CoreIndicators
import numpy as np


class ATRBands(CoreIndicators):
    """
    ATR Bands indicator.

    ATR Bands are volatility-based bands that use the Average True Range (ATR) to 
    set the distance between the upper and lower bands from a moving average of 
    the price. These bands can help traders identify potential overbought or 
    oversold conditions in the market.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Array of historical price data.
    - n (int): Number of periods for the ATR calculation.
    - k (float): Multiplier for the ATR to set the distance of the bands.

    Returns:
    np.ndarray: Array containing the lower and upper bands.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, atr, n, k):
        dimension = self._extractor.extract_dimension(timeseries)
        upper     = dimension + k * atr
        lower     = dimension - k * atr
        dimension = np.diff(dimension)
        dimension = np.hstack(((np.mean(dimension)), dimension))
        upper     = self.stabilize(dimension * -1, upper, np.minimum)
        lower     = self.stabilize(dimension, lower, np.maximum)
        upper     = np.roll(upper, -1)
        lower     = np.roll(lower, -1)
        upper[0]  = lower[0] = np.nan
        return np.array((lower, upper))


    def stabilize(self, dimension, band, function):
        return np.where(0 < dimension,
                function(band, np.roll(band, 1)),
                band)
