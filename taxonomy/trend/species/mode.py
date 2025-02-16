from indicators.base.core import CoreIndicators
from collections import Counter
import numpy as np


class Mode(CoreIndicators):
    """
    Mode Indicator.


    Computes the n-period rolling mode of a timeseries.

    Parameters:
    timeseries (pd.DataFrame, or Stock): The input financial time series.
    n (int): The period of the mode, indicating the number of data points to include in the calculation.
    dimension (str): The target column in the timeseries to use for calculation.

    Returns:
    np.ndarray: The Rolling Mode values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        return self.mode(dimension, n)


    def mode(self, dimension, n):
        mode = np.full(dimension.shape[0], np.nan)
        for i in range(n, dimension.shape[0]):
            subset  = dimension[i - n: i + 1]
            counter = Counter(subset)
            mode[i] = counter.most_common(1)[0][0]
        return mode


