from indicators.base.core import CoreIndicators


class BalanceOfPower(CoreIndicators):
    """
    Balance of Power (BOP) Indicator.

    The Balance of Power (BOP) indicator helps predict market trends by evaluating the 
    strength of buyers and sellers. It assesses who has more influence over price movements 
    by comparing the relationship between the opening, closing, high, and low prices. BOP 
    values can indicate potential shifts in market sentiment, offering insights into possible 
    future price movements.

    Parameters:
    - timeseries (pd.DataFrame): The input financial time series.
    - smoothing (int): The period for smoothing the BOP values. Set to 0 for no smoothing.
    - mode (str): The smoothing mode. Options: "sma" (Simple Moving Average), "ema" (Exponential 
    Moving Average), "tema" (Triple Exponential Moving Average), "wilders" (Wilder's Exponential 
    Moving Average).

    Returns:
    - numpy.ndarray: The scaled Balance of Power values in the range [-1, 1].
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, smoothing, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        matrix = self._extractor.extract_matrix(timeseries)
        open, high, low, close = matrix[:,1], matrix[:,2], matrix[:,3], matrix[:,4]
        bop = self._tools.divide(close - open, high - low)
        if smoothing: bop = getattr(self, f"_{mode}")(bop, smoothing)
        scaler = self._tools.sklearn_minmaxscaler(bop, -1, 1)
        bop    = scaler.transform(bop.reshape(-1, 1)).flatten()
        return bop
