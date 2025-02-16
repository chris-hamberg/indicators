from indicators.base.core import CoreIndicators
import numpy as np


class ParabolicSAR(CoreIndicators):
    """
    Parabolic Stop and Reverse (SAR) indicator.

    The Parabolic SAR (Stop and Reverse) indicator is used to determine the 
    direction of a security's price movement and generate potential entry and exit 
    points. It is used to identify potential reversals in the price direction of 
    an asset.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Array of historical price data.
    - AF (float): Acceleration factor for the SAR calculation.
    - increment (float): Increment for the acceleration factor.
    - limit (float): Upper limit for the acceleration factor.

    Returns:
    np.ndarray: Array containing the Parabolic SAR values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, AF, increment, limit):

        matrix = self._extractor.extract_matrix(timeseries)

        state  = "falling"
        EP     = matrix[0, 2]
        sar    = np.zeros(matrix.shape[0])
        sar[0] = EP

        for e in range(1, matrix.shape[0]):
            low, high = matrix[e, 3], matrix[e, 2]
            pSAR = sar[e-1]

            # Rising SAR
            if state == "rising":
                SAR = pSAR + AF * (EP - pSAR)
                AF  = min(AF + increment, limit) if (EP < high) else AF
                EP  = max(high, EP)
                switch = low < SAR
                alpha, beta = "falling", "rising"

            # Falling SAR
            else:
                SAR = pSAR - AF * (pSAR - EP)
                AF  = min(AF + increment, limit) if (low < EP) else AF
                EP  = min(low, EP)
                switch = SAR < high
                alpha, beta = "rising", "falling"

            # Reversal
            sar[e] = EP if switch else SAR
            AF = 0.02 if switch else AF
            state = alpha if switch else beta

        return sar
