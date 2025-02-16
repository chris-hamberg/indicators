from statsmodels.stats.diagnostic import het_arch
from indicators.base.core import CoreIndicators
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
import numpy as np


class Heteroscedasticity(CoreIndicators):
    """
    Heteroscedasticity test for financial time series.

    The Heteroscedasticity class provides a test for heteroscedasticity in financial
    time series using the ARCH-LM test. It can be used to determine if the variance
    of the residuals from a model is dependent on the level of the series.

    Parameters:
    - model: The model object for which to test heteroscedasticity (accepts timeseries)

    Returns:
    bool: True if the null hypothesis of homoscedasticity is rejected (i.e., there
    is evidence of heteroscedasticity), False otherwise.
    """
    def __init__(self, pthreshold=0.05):
        super().__init__()
        self._pthreshold = pthreshold


    def __call__(self, model):
        try: residuals = model.resid
        except AttributeError:
            residuals = self.synthesize(model)
        pvalue = het_arch(residuals)[1]
        return pvalue < self._pthreshold

    
    def synthesize(self, model):
        model, LogR = self.mock(model)
        if hasattr(model, "resid"):
            residuals = model.resid
        elif hasattr(model, "resids"):
            residuals = model.resids(LogR)
            residuals = np.where(residuals != 0, residuals, 1e-20)
        else:
            residuals = np.array([1e-20])
        return residuals


    def mock(self, model):
        scaler = MinMaxScaler(feature_range=(1, 100))
        LogR   = scaler.fit_transform(model.reshape(-1, 1)).flatten()
        model  = arch_model(LogR, vol="GARCH", p=1, q=1, dist="t")
        model.fit(disp="off")
        return model, LogR
