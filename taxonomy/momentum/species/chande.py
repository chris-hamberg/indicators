from indicators.base.core import CoreIndicators
import numpy as np


class ChandeMomentum(CoreIndicators):
    """
    Chande Momentum Oscillator (CMO) indicator.

    The Chande Momentum Oscillator (CMO) is a technical indicator that measures 
    the momentum of a security's price relative to the recent price changes. It is 
    used to identify overbought and oversold conditions and potential trend 
    reversals.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for the calculation.
    - dimension (str): Dimension of the timeseries to consider, e.g., "close" or "high".

    Returns:
    np.ndarray: Array containing the CMO values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        cmo               = np.full(dimension.shape[0], np.nan)
        differences       = (dimension - np.roll(dimension, 1))[1:]
        kronecker_delta   =  np.where(0 < differences, differences, 0)
        kronecker_epsilon = -np.where(differences < 0, differences, 0)
        kappa             = self._summation(kronecker_delta,   n)
        mu                = self._summation(kronecker_epsilon, n)
        cmo[1:]           = self._tools.divide(kappa - mu, kappa + mu) * 100
        return cmo
