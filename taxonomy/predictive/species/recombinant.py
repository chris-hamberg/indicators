from indicators.base.core import CoreIndicators
from sklearn.metrics import r2_score
import numpy as np


class RecombinantR(CoreIndicators):
    """
    Attempts to remove the n-period lag from SMA.

    Computes the SMA, and fits an n-degree (3) regression. Extrapolates the 
    regression n-periods into the future, and shifts both the regression and 
    SMA back n-periods. The missing future n-periods in the SMA are imputed with
    SMA[-1]. Finally, the transformed SMA, regression, and actual price are 
    combined as a weighted function representing the "unlagged" n-period SMA
    estimate.

    WARNING: Suffers from look ahead bias
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, degree, s_weight, r_weight, 
            d_weight):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        
        s, r, d = self.normalize(s_weight, r_weight, d_weight)

        # Compute the SMA
        sma = self._sma(dimension, n)
        idx = np.argmax(~np.isnan(sma))

        # Compute the regression, impute & shift the SMA
        regression = self.fit(dimension, sma, idx, degree)
        sma        = self.impute(sma, idx)

        # Recombine the 3 datasets
        recombinantR = (sma * s) + (regression * r) + (dimension * d)
        
        # Measure the result
        r2 = r2_score(sma[:-idx], recombinantR[:-idx])

        return recombinantR, round(r2 * 100, 2)

        
    def impute(self, sma, idx):

        # Shift and Impute SMA
        shifted           = sma[idx:]
        impute_val        = sma[-1]
        number_of_missing = idx
        imputed_array     = number_of_missing * [impute_val]

        imputed = np.hstack((shifted, imputed_array))

        return imputed


    def fit(self, dimension, sma, idx, degree):
        
        # Train Test Split
        X_train = np.arange(dimension[idx:].shape[0])
        X_test  = np.arange(dimension.shape[0] + idx)
        Y_train = sma[idx:]

        # Fit | Predict
        coeffs    = np.polyfit(X_train, Y_train, degree)
        y_predict = np.polyval(coeffs, X_test)
        y_predict = y_predict[idx:]

        return y_predict


    def normalize(self, s, r, d):

        # Normalize the recombination weights
        total = s + r + d
        s /= total
        r /= total
        d /= total
        return s, r, d
