from indicators.taxonomy.momentum.core.obv import CoreOBV
from indicators.base.core import CoreIndicators


class OnBalancedVolume(CoreIndicators):
    """
    On-Balance Volume (OBV) indicator with support for volume weighting and normalization.

    On-Balance Volume (OBV) is a technical indicator that uses volume flow to 
    predict changes in stock price. It measures buying and selling pressure by 
    adding the volume on up days and subtracting volume on down days. This 
    implementation supports weighting by volume and normalization of the OBV values.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - weight (float): Weighting factor for volume, ranging from 0.0 to 1.0.
    - normalize (bool): Flag indicating whether to normalize the OBV values.

    Returns:
    np.ndarray: Array containing the OBV values.
    """
    def __init__(self):
        super().__init__()
        self.core = CoreOBV()


    def __call__(self, timeseries, weight, normalize):
        matrix = self._extractor.extract_matrix(timeseries)
        obv, _ = self.core.compute(matrix, weight, mode="flat")
        return obv if not normalize else self._tools.rescale(obv)
