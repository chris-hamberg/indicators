from indicators.base.core import CoreIndicators
from indicators.stock import Stock
import numpy as np


class PriceWeightedVolume(CoreIndicators):
    """
    The Price-Weighted Volume Indicator 

    This is a macro indicator giving a high-level normalized view on market 
    sentiment, and investor interested based on computing out of how many 
    dollars available for trading stocks, market-wide in total, are allocated to 
    the particular stock.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input time series data. Must be 
    in 1-min timescale.
    - n (int): The lookback period to smooth the Price-Weighted Volume by. The 
    default is 10 (days).

    Returns:
    - numpy.ndarray: An array of normalized ratios representing market sentiment.

    Example:
    ```python
    pwv = ix.sentiment.pwv(timeseries, n=5)
    ```
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        assert timeseries.shape[1] == 6, "Must be 2-dimensional timeseries."
        if not isinstance(timeseries, Stock): 
            timeseries = Stock(matrix=timeseries)
        assert timeseries.timescale == 1, "Must be at 1-min data granularity."
        pwv = self.aggregate(timeseries)
        return pwv if not n else self._smm(pwv, n)


    def aggregate(self, stock):
        timestamps = stock.timestamps.astype("datetime64[D]")
        dates, price, volume = np.unique(timestamps), [], []
        for date in dates:
            open  = date.astype("datetime64[m]") + np.timedelta64(570, "m")
            close = open + np.timedelta64(390, "m")
            chart = stock[(open <= stock.timestamps) & (stock.timestamps <= close)]
            price.append(np.nanmedian(chart[:,4]))
            volume.append(np.nansum(chart[:,5]))
        return np.array(price) * np.array(volume)
