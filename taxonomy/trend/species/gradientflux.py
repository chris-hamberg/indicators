from indicators.base.core import CoreIndicators
import numpy as np


class GradientFlux(CoreIndicators):
    """
    Gradient Flux Indicator.

    The Gradient Flux Indicator computes the average of time scales, providing insights
    into the gradient changes in a financial time series. It can be used to identify 
    trends and changes in trend direction.

    Parameters:
    - timeseries (Stock): The input financial time series.
    - n (int): The number of periods over which to calculate the Gradient Flux.
    - dimension (str): The dimension of interest in the time series, e.g., 'close' for 
      closing prices.
    - mode (str): The computation mode, which determines the weighting scheme. Options 
      are 'linear', 'weighted', and 'fast'.

    Returns:
    - numpy.ndarray: An array containing the computed Gradient Flux values.
    """
    modes = {
            "linear"  : {
        "index": 1, "omega": lambda n: np.ones(n),      "factor": 5},
            "weighted": {
        "index": 1, "omega": lambda n: np.arange(1, n), "factor": 5},
            "fast"    : {
        "index": 2, "omega": lambda n: np.arange(1, n), "factor": 1}}


    def __init__(self):
        super().__init__()
        self.error = f"Mode must be one of: {list(GradientFlux.modes.keys())}"


    def __call__(self, timeseries, n, dimension, mode):
        stock = self._extractor.convert_to_stock(timeseries)
        if not mode in GradientFlux.modes: raise ValueError(self.error)
        first, gradientFlux = stock[dimension][0], np.zeros(stock.shape[0])
        index     = GradientFlux.modes[mode]["index"]
        factor    = GradientFlux.modes[mode]["factor"]
        omega     = GradientFlux.modes[mode]["omega"](n)
        dimension = self._extractor._dimensions.index(dimension)
        for i in range(index, n):
            gradientFlux += stock.scale(i*factor)[:,dimension] * omega[i-index]
        gradientFlux /= omega.sum()
        drift = first - gradientFlux[0]
        if not np.isnan(drift): return gradientFlux + drift
        return gradientFlux
