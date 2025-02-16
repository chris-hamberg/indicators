from indicators.base.core import CoreIndicators


class BBandW(CoreIndicators):
    """
    Bollinger Band Width (BBandW) indicator.

    Bollinger Band Width is an indicator derived from Bollinger Bands that measures 
    the width of the bands. It can help identify periods of high or low volatility.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
    should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the moving average.
    - k (float): The number of standard deviations to add or subtract from the 
    moving average to calculate the upper and lower bands.
    - dimension (str): The column name representing the dimension in the 
    timeseries. Typically, 'close' is used for this parameter.

    Returns:
    numpy.ndarray: An array containing the calculated Bollinger Band Width values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, bands):
        lower, mid, upper = bands
        dividend = upper - lower
        width    = self._tools.divide(dividend, mid)
        return width
