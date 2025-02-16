from indicators.taxonomy.volatility.core.garch import GARCH as Model
from indicators.base.core import CoreIndicators
from collections import namedtuple
from scipy.stats import pearsonr
import numpy as np


Volatility = namedtuple("Volatility", ["stationarity", "heteroscedasticity",
    "volatility", "forecast", "confidence", "accuracy", "pvalue"])


class GARCH(CoreIndicators):
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model indicator.

    The GARCH indicator uses a GARCH model to analyze volatility in financial time series data. 
    It provides measures of stationarity, heteroscedasticity, volatility, and forecasts future 
    volatility based on the GARCH model.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which should include 
    columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the GARCH analysis.
    - dimension (str): The column name representing the dimension in the timeseries. This could 
    be 'close' for price data or 'volume' for volume data.
    - sims (int): The number of simulations to run for the GARCH model.
    - LogR (bool): Indicates whether to use logarithmic returns.

    Returns:
    collections.namedtuple: A named tuple containing the results of the GARCH analysis, including 
    information about stationarity, heteroscedasticity, volatility, forecast, confidence, accuracy, 
    and p-value.
    """
    def __init__(self):
        super().__init__()
        self.model = Model()


    def __call__(self, timeseries, n, dimension, sims, LogR):
        if LogR: 
            dimension = self._LogR(timeseries, dimension)
        else:    
            dimension = self._extractor.extract_dimension(timeseries, dimension)
        results = self.model.predict(dimension, n, sims)
        variance = self.measure(dimension, results, n)
        self.forecast(dimension, variance, results)
        return Volatility(**results)


    def measure(self, dimension, results, n):
        variance   = self._variance(dimension, n)
        index      = np.argmax(~np.isnan(variance))
        actual     = variance[index:]
        estimate   = results["volatility_model"][index:]
        pcc, pval  = pearsonr(actual, estimate)
        results["volatility"] = np.sqrt(variance)
        results["accuracy"]   = pcc
        results["pvalue"]     = pval
        results.pop("volatility_model")
        return variance


    def sqrt(self, X):
        return np.where(0 <= X, np.sqrt(X), np.nan)


    def forecast(self, dimension, variance, results):
        factor = results["forecast"][0] / variance[-1]
        results["forecast"] /= factor
        results["forecast"] = self.sqrt(results["forecast"])
        pad = np.full(dimension.shape[0] - 1, np.nan)
        results["forecast"] = np.hstack((pad, results["forecast"]))
