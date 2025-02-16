from indicators.base.core import CoreIndicators
from statsmodels.tsa.stattools import adfuller


class Stationarity(CoreIndicators):
    """
    Test for stationarity in a financial time series.

    Stationarity refers to the property of having a constant mean and variance over time.
    A stationary time series is important in many statistical models. This test uses the
    Augmented Dickey-Fuller test (ADF) to determine stationarity.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - dimension (str): The column name representing the dimension in the timeseries.

    Returns:
    bool: True if the time series is stationary, False otherwise.
    """
    def __init__(self, pthreshold=0.05):
        super().__init__()
        self._pthreshold = pthreshold


    def __call__(self, timeseries, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        pvalue = adfuller(dimension)[1]
        return pvalue < self._pthreshold
