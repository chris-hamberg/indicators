from indicators.taxonomy.pulsewave.core.generic import Generic
from indicators.base.core import CoreIndicators
from scipy.stats import kurtosis


class Kurtosis(CoreIndicators):
    """
    Kurtosis of a financial time series.

    Computes the kurtosis of a financial time series. Kurtosis measures the 
    "tailedness" of the probability distribution of a real-valued 
    random variable. It is a descriptor of the shape of the probability distribution.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - n (int): The number of periods over which to compute the kurtosis.
    - dimension (str): The column name representing the dimension in the timeseries.
    - mode (str): The mode of computation. Options are "flat" (the entire dataset), 
    "rolling", or "expanding".

    Returns:
    np.array: An array of computed kurtosis values, aligned with the input timeseries.
    """
    def __init__(self):
        super().__init__()
        self.processor = Generic()


    def __call__(self, timeseries, n, dimension, mode):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        return self.processor.compute(dimension, n, kurtosis, mode)
