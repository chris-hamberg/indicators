from indicators.base.core import CoreIndicators
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from collections import namedtuple
import numpy as np


Forecast = namedtuple("Forecast", ["forecast", "r2", "mse", "error"])


class MCForecast(CoreIndicators):
    """
    Monte Carlo Forecasting for financial time series.

    The Monte Carlo Forecasting method uses simulations to predict future prices 
    for a financial time series. It combines volatility and trend models to generate 
    multiple possible outcomes, which can be used to estimate the most likely future 
    price trajectory.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Array of historical price data.
    - horizon (int): Number of periods to forecast into the future.
    - lookback (int): Number of periods to look back for model calculations.
    - dimension (str): The data dimension to use for forecasting.
    - sims (int): Number of simulations to run for forecasting.
    - method (str): Method for combining simulation outcomes, either "mean" or "median".
    - test (bool): Flag indicating whether to evaluate the forecast accuracy.

    Returns:
    Forecast: A namedtuple containing the forecasted prices, R-squared score, mean 
    squared error, and percentage error (if applicable).
    """
    _methods = ["mean", "median"]


    def __init__(self):
        super().__init__()
        self.error = f"Method must be one of: {MCForecast._methods}"


    def __call__(self, timeseries, horizon, lookback, dimension, sims, method, 
            test):
        
        if method not in MCForecast._methods: raise ValueError(self.error)

        args = [timeseries, horizon, dimension, sims, lookback, method]

        if test: r2, mse, perc = self.measure(*args)
        else: r2 = mse = perc = np.nan

        # Forecast the price.
        forecast = self.monte_carlo(*args)
        forecast = Forecast(forecast, r2, mse, perc)

        return forecast


    def monte_carlo(self, timeseries, horizon, dimension, sims, lookback, method):
        
        # Construct Volatility, Trend, and Price models.
        args = [timeseries, horizon, dimension, lookback]
        volatility_model, trend_model, price_model = self.model(*args)
        
        # Construct Simulations based on the models.
        args = [horizon, sims, volatility_model, trend_model, price_model]
        simulations = self.simulate(*args)

        # Find the most likely outcome.
        if   method == "median": predictions = np.median(simulations, axis=0)
        elif method == "mean":   predictions = np.mean(  simulations, axis=0)
        pad         = np.full(timeseries.shape[0] - 1, np.nan)
        predictions = np.hstack((pad, (price_model,), predictions))
        
        return predictions


    def measure(self, timeseries, horizon, dimension, sims, lookback, method):
        X_train = timeseries[:-horizon]
        pred    = self.monte_carlo(X_train, horizon, dimension, sims, lookback, 
                method)
        index   = np.argmax(~np.isnan(pred))
        pred    = pred[index:]
        X_test  = timeseries[-horizon:]
        X_test  = self._extractor.extract_dimension(X_test, dimension)
        r2      = r2_score(X_test, pred)
        mse     = mean_squared_error(X_test, pred)
        perc    = round(mse / X_test.mean() * 100, 2)
        return r2, mse, perc


    def simulate(self, horizon, sims, volatility_model, trend_model, price_model):
        shape = horizon, sims
        if isinstance(volatility_model, np.ndarray):
            volatility_model = volatility_model[:, np.newaxis]
        if isinstance(trend_model, np.ndarray):
            trend_model = trend_model[:, np.newaxis]
        shocks      = np.random.normal(trend_model, volatility_model, shape)
        simulations = np.cumprod(1 + shocks.T, axis=1) * price_model
        return simulations


    def model(self, timeseries, horizon, dimension, lookback):
        dimension        = self._extractor.extract_dimension(timeseries, dimension)
        returns, price   = self._roc(dimension, 1) / 100, dimension[-1]
        mean             = self._sma(returns, lookback)
        variance         = self._variance(returns, lookback)
        trend_model      = mean - (variance * 0.5)
        trend_model      = trend_model[-horizon:]
        volatility_model = np.sqrt(variance)[-horizon:]
        return volatility_model, trend_model, price
