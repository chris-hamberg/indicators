from indicators.base.core import CoreIndicators


class PhiTensor(CoreIndicators):
    """
    PhiTensor indicator.

    This indicator is similar to the Gradient Flux but uses time scale intervals 
    separated by distances equivalent to Fibonacci numbers. It serves as a powerful 
    smoothing and trend indicator, akin to other moving averages.

    Parameters:
    - timeseries (Stock): The input financial time series. This indicator requires 
    the input to be of type Stock, and pd.DataFrame is not accepted.
    - n (int): The number of Fibonacci numbers to use in the computation.
    - dimension (str): The column name representing the dimension in the timeseries.

    Returns:
    np.ndarray: The PhiTensor values representing the smoothed and trended data.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, _cls):
        stock = self._extractor.convert_to_stock(timeseries)
        if len(_cls._fibonacci) != n:
            _cls._update_fibonacci(self, n)
        phiTensor = stock[dimension].copy()
        dimension = self._extractor._dimensions.index(dimension)
        for i in _cls._fibonacci:
            phiTensor += stock.scale(i)[:,dimension]
        return phiTensor / (_cls._fibonacci.shape[0] + 1)
