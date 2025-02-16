from indicators.taxonomy.pulsewave.core.generic import Generic
from indicators.base.core import CoreIndicators
from scipy.stats import skew


class Skew(CoreIndicators):
    """
    Skewness for financial time series.

    Skewness measures the asymmetry of the probability distribution of a 
    real-valued random variable. In the context of financial markets, skewness 
    can provide insights into the potential for extreme returns in one direction 
    compared to the other.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - n (int): The number of periods to consider in the calculation.
    - dimension (str): The column name of interest in the time series.
    - mode (str): The computation mode: "flat", "rolling", or "expanding".

    Returns:
    - numpy.ndarray: An array containing the computed skewness values.
    """
    def __init__(self):
        super().__init__()
        self.processor = Generic()


    def __call__(self, timeseries, n, dimension, mode):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        return self.processor.compute(dimension, n, skew, mode)
