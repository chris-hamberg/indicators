from indicators.base.core import CoreIndicators
from scipy.stats import zscore
import numpy as np


class Clusters(CoreIndicators):
    """
    Clusters indicator.

    The Clusters indicator identifies clusters of extreme values in a given dimension 
    of a timeseries based on z-scores. It can help identify periods of unusual activity 
    or outliers.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
      should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - k (float): The number of standard deviations from the mean to consider as a threshold 
      for identifying clusters.
    - dimension (str): The column name representing the dimension in the 
      timeseries. This could be 'close' for price data or 'volume' for volume data.

    Returns:
    numpy.ndarray: An array containing the values of the input dimension that fall 
    within the identified clusters, with outliers replaced by NaN.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, k, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        matrix          = self._extractor.extract_matrix(timeseries)
        zscores         = np.abs(zscore(dimension))
        upper_threshold = zscores.mean() + k * zscores.std()
        lower_threshold = zscores.mean() - k * zscores.std()
        condition1      = zscores < lower_threshold
        condition2      = upper_threshold < zscores
        clusters = np.where((condition1) | (condition2), matrix[:,4], np.nan)
        return clusters
