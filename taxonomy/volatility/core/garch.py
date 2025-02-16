from indicators.taxonomy.pulsewave.interface import Pulsewave
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import numpy as np


class GARCH:


    def __init__(self, pthreshold=0.05, scaling_factor=1000):
        self.pulsewave      = Pulsewave()
        self.pthreshold     = pthreshold
        self.scaling_factor = scaling_factor
        self.keys = ("stationarity", "heteroscedasticity", "volatility_model",
                "forecast", "confidence")


    def predict(self, LogR, horizon, sims):
        params = {"horizon":horizon + 1, "method":"simulation", "simulations":sims}
        default = dict.fromkeys(self.keys, np.nan)
        hetero, default["stationarity"] = True, False
        if not self.pulsewave.stationarity(LogR): return default
        results, LogR = default, LogR * self.scaling_factor
        model = self.build(LogR, "GARCH")
        if not self.pulsewave.heteroscedasticity(model): 
            model, hetero = self.build(LogR, "ARCH"), False
        results["confidence"]         = self.confidence(model)
        results["forecast"]           = self.forecast(  model, **params)
        results["volatility_model"]   = self.volatility(model)
        results["heteroscedasticity"] = hetero
        results["stationarity"]       = True
        return results


    def confidence(self, model):
        residuals   = model.resid
        confidence  = acorr_ljungbox(residuals)
        cardinality = confidence.index.shape[0]
        evidence    = confidence[confidence.lb_pvalue >= self.pthreshold]
        evidence    = evidence.index.shape[0]
        confidence  = evidence / cardinality * 100
        confidence  = min(confidence, 95.0)
        confidence  = round(confidence, 2)
        return confidence


    def forecast(self, model, **params):
        forecast = model.forecast(**params)
        forecast = forecast.variance.values[0]
        forecast = forecast
        return forecast


    def volatility(self, model):
        volatility = model.conditional_volatility
        volatility = volatility
        return volatility


    def build(self, LogR, model="GARCH"):
        params = {"vol": model, "p": 1, "dist": "t"}
        if model == "GARCH": params.update({"q": 1})
        model = arch_model(LogR, **params)
        model = model.fit(disp="off")
        return model
