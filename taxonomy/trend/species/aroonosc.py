from indicators.base.core import CoreIndicators


class AROONOscillator(CoreIndicators):
    """
    Aroon Oscillator Indicator.

    Aroon Oscillator is a leading indicator that measures the difference between 
    Aroon Up and Aroon Down.
    - A positive value indicates a bullish trend.
    - A negative value indicates a bearish trend.
    - Values around zero indicate a period of consolidation or indecision in the market.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - n (int): The number of periods to consider for calculating the Aroon indicator.

    Returns:
    - numpy.ndarray: A 1D NumPy array containing the Aroon Oscillator values for each period 
    in the input time series. Each value represents the difference between Aroon Up and Aroon 
    Down for that period.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, up, down):
        return up - down
