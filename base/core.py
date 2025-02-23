from indicators.utils.errors import LookbackException
from indicators.base.extractor import Extractor
from indicators.base.adaptive import Adaptive
from indicators.utils.windows import Windows
from indicators.utils.tools import Tools
import numpy as np
import warnings


class CoreIndicators(Adaptive):


    _fibonacci = ()
    _retracements = np.array((0.236, 0.382, 0.5, 0.618, 0.764, 1))
    _modes        = ["sma", "ema", "tema", "wilders"]


    @property
    def cluster(self):
        s = [s for s in self.__class__.__dict__.keys() if "_" not in s]
        return s


    def __init__(self):
        super().__init__()
        self._error     = f"Mode must be one of: {CoreIndicators._modes}"
        self._extractor = Extractor()
        self._windows   = Windows()
        self._tools     = Tools()


    def __repr__(self):
        taxa = name = self.__class__.__name__
        taxa   = f"{taxa} Group"
        border = "-" * len(taxa)
        cluster = ""
        for species in self.cluster:
            cluster += f"    {species}\n"
        header = f"\n  {taxa}\n  {border}\n"
        body   = f"{cluster}  {border}\n"
        note   = f"  See >>> help(object.{name.lower()}) for details."
        return f"{header}{body}{note}"


    def _summation(self, timeseries, n=14, dimension="Close"):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        summation = np.full(dimension.shape[0], np.nan)
        summation[n - 1:] = np.convolve(dimension, np.ones(n), mode="valid")
        return summation


    def _minima(self, timeseries, n=14, dimension="Close"):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        rolling, minima = self._windows.rolling(dimension, n)
        try: minima[n - 1:]  = np.nanmin(rolling, axis=1)
        except np.exceptions.AxisError: return minima
        return minima


    def _maxima(self, timeseries, n=14, dimension="Close"):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        rolling, maxima = self._windows.rolling(dimension, n)
        try: maxima[n - 1:]  = np.nanmax(rolling, axis=1)
        except np.exceptions.AxisError: return maxima
        return maxima
        

    def _variance(self, timeseries, n=14, dimension="Close"):
        """
        Calculates the variance of a given dimension in a financial time series, 
        which can provide insights into the volatility of the market.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series, 
          which should include columns for 'open', 'high', 'low', 'close', and 
          'volume'.
        - n (int, optional): The number of periods to consider for calculating 
          the variance. Defaults to 14.
        - dimension (str, optional): The column name representing the dimension 
          in the timeseries for which to calculate the variance. Typically, 'close' 
          is used for this parameter. Defaults to "Close".

        Returns:
        numpy.ndarray: An array containing the calculated variances, providing 
        insights into the market's volatility.
        """
        dimension  = self._extractor.extract_dimension(timeseries, dimension)
        rolling, _ = self._windows.rolling(dimension, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try: variance = np.nanvar(rolling, axis=1)
            except np.exceptions.AxisError:
                variance = np.zeros(dimension.shape[0])
        variance = self._tools.resize(dimension, variance)
        return variance


    def _std(self, timeseries, n=14, dimension="Close"):
        """
        Calculates the standard deviation of a given dimension in a financial 
        time series, which can provide insights into the volatility of the market.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series, 
          which should include columns for 'open', 'high', 'low', 'close', and 
          'volume'.
        - n (int, optional): The number of periods to consider for calculating 
          the standard deviation. Defaults to 14.
        - dimension (str, optional): The column name representing the dimension 
          in the timeseries for which to calculate the standard deviation. Typically, 
          'close' is used for this parameter. Defaults to "Close".

        Returns:
        numpy.ndarray: An array containing the calculated standard deviations, 
        providing insights into the market's volatility.
        """
        return np.sqrt(self._variance(timeseries, n, dimension))


    def _roc(self, timeseries, n=10, dimension="Close"):
        """
        Calculates the Rate of Change (ROC) of a given dimension in a financial 
        time series, which indicates the percentage change from one period to 
        another.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series, 
          which should include columns for 'open', 'high', 'low', 'close', and 
          'volume'.
        - n (int, optional): The number of periods to consider for calculating 
          the ROC. Defaults to 10.
        - dimension (str, optional): The column name representing the dimension 
          in the timeseries for which to calculate the ROC. Typically, 'close' 
          is used for this parameter. Defaults to "Close".

        Returns:
        numpy.ndarray: An array containing the calculated ROC values, indicating 
        the percentage change from one period to another.
        """
        dimension    = self._extractor.extract_dimension(timeseries, dimension)
        anterior     = np.full(dimension.shape[0], np.nan)
        anterior[n:] = np.roll(dimension, n)[n:]
        roc          = self._tools.divide(dimension - anterior, anterior) * 100
        return roc


    def _LogR(self, timeseries, dimension="Close"):
        """
        Calculates the logarithmic return (LogR) of a given dimension in a 
        financial time series, which represents the logarithm of the ratio of 
        the current value to the previous value.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series, 
          which should include columns for 'open', 'high', 'low', 'close', and 
          'volume'.
        - dimension (str, optional): The column name representing the dimension 
          in the timeseries for which to calculate the LogR. Typically, 'close' 
          is used for this parameter. Defaults to "Close".

        Returns:
        numpy.ndarray: An array containing the calculated LogR values, 
        representing the logarithm of the ratio of the current value to the 
        previous value.
        """
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        LogR = np.log(dimension) - np.log(np.roll(dimension, 1))
        return np.hstack(((0,), LogR[1:]))


    def _inverseLogR(self, LogR, initial_price):
        """
        Calculates the inverse of the logarithmic return (LogR) to reconstruct 
        price values from LogR values.

        Parameters:
        - LogR (numpy.ndarray): An array containing the LogR values.
        - initial_price (float): The initial price value to start the 
          reconstruction.

        Returns:
        numpy.ndarray: An array containing the reconstructed price values.
        """
        prices = np.zeros_like(LogR)
        prices[0] = initial_price
        for i in range(1, LogR.shape[0]):
            prices[i] = prices[i - 1] * np.exp(LogR[i])
        return prices


    def _sma(self, timeseries, n=14, dimension="Close", adaptive=False):
        """
        Simple Moving Average (SMA)

        Calculates the Simple Moving Average (SMA) of a given timeseries.
        SMA is a commonly used technical indicator that smooths out price data
        by creating a constantly updated average price. It is useful for traders
        to identify the direction of a trend or to determine support and resistance levels.
        If adaptive is True, calculates the Adaptive SMA based on the Average True Range (ATR).

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the moving average.
        - dimension (str): The column name representing the dimension in the 
        timeseries. Typically, 'close' is used for this parameter.
        - adaptive (bool): If True, calculates the Adaptive SMA based on the ATR. Default is False.

        Returns:
        numpy.ndarray: An array containing the SMA values.
        """
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        if adaptive:
            matrix  = self._extractor.extract_matrix(timeseries)
            vectors = self._vectors(matrix, n)
            sma     = self._vector_space(dimension, vectors, np.mean)
        else:
            sma = np.convolve(dimension, np.ones(n)/n, mode="valid")
            sma = self._tools.resize(dimension, sma)
        return sma

    
    def _smm(self, timeseries, n=14, dimension="Close"):
        """
        Simple Moving Median (SMM)

        Similar to SMA, but instead of the mean, it computes the median.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the moving median.
        - dimension (str): The column name representing the dimension in the 
        timeseries. Typically, 'close' is used for this parameter.

        Returns:
        numpy.ndarray: An array containing the SMM values.
        """
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        smm = self._windows.rolling_function(dimension, n, np.nanmedian)
        return smm


    def _ema(self, timeseries, n=14, dimension="Close", adaptive=False, 
            _temaX=None):
        """
        Exponential Moving Average (EMA)

        Calculates the Exponential Moving Average (EMA) of a given timeseries.
        EMA is a type of moving average that gives more weight to recent data points,
        making it more responsive to recent price changes. It is useful for traders
        to identify the direction of a trend or to determine support and resistance levels.
        If adaptive is True, calculates the Adaptive EMA based on the Average True Range (ATR).

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the moving average.
        - dimension (str): The column name representing the dimension in the 
        timeseries. Typically, 'close' is used for this parameter.
        - adaptive (bool): If True, calculates the Adaptive EMA based on the ATR. Default is False.

        Returns:
        numpy.ndarray: An array containing the EMA values.
        """
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        if _temaX is not None: dimension = _temaX
        index     = np.argmax(~np.isnan(dimension))
        ema0      = np.nanmean(dimension[index:index+n])
        ema       = np.hstack((np.full(index, ema0), dimension[index:]))
        if adaptive: matrix = self._extractor.extract_matrix(timeseries)
        vectors = np.full_like(ema,n) if not adaptive else self._vectors(matrix,n)
        ema     = self._ema_vector_space(ema, n, vectors, ema)
        return ema


    def _tema(self, timeseries, n=14, dimension="Close", adaptive=False):
        """
        Triple Exponential Moving Average (TEMA)

        Calculates the Triple Exponential Moving Average (TEMA) of a given timeseries.
        TEMA is a type of moving average that applies a triple smoothing technique to
        the EMA values, providing a smoother trend indication. It is useful for traders
        to identify the direction of a trend or to determine support and resistance levels.
        If adaptive is True, calculates the Adaptive TEMA based on the Average True Range (ATR).

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the moving average.
        - dimension (str): The column name representing the dimension in the 
        timeseries. Typically, 'close' is used for this parameter.
        - adaptive (bool): If True, calculates the Adaptive TEMA based on the ATR. Default is False.

        Returns:
        numpy.ndarray: An array containing the TEMA values.
        """
        ema1 = self._ema(timeseries, n, dimension, adaptive)
        ema2 = self._ema(timeseries, n, dimension, adaptive, ema1)
        ema3 = self._ema(timeseries, n, dimension, adaptive, ema2)
        return 3 * (ema1 - ema2) + ema3


    def _wilders(self, timeseries, n=14, dimension="Close"):
        """
        Wilder's Exponential Moving Average (EMA)

        Calculates the Exponential Moving Average (EMA) using J. Welles Wilder's
        smoothing technique. Wilder's EMA is similar to a standard EMA but uses a
        different smoothing formula. It is useful for traders to identify the direction
        of a trend or to determine support and resistance levels.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the moving average.
        - dimension (str): The column name representing the dimension in the 
        timeseries. Typically, 'close' is used for this parameter.

        Returns:
        numpy.ndarray: An array containing the Wilder's EMA values.
        """
        dimension  = self._extractor.extract_dimension(timeseries, dimension)
        wilders    = np.full(dimension.shape[0], np.nan, dtype=float)
        index      = np.argmax(~np.isnan(dimension))
        try: wilders[index + n] = np.nanmean(dimension[index:index+n])
        except IndexError: return wilders
        for i in range(index + n + 1, dimension.shape[0]):
            wilders[i] = (wilders[i - 1] * (n - 1) + dimension[i]) / n
        return wilders
    

    def _tr(self, timeseries):
        """
        True Range (TR) indicator.

        True Range is often used as part of the Average True Range (ATR) indicator, 
        which helps traders identify potential market trends and assess the 
        likelihood of price movements. A higher True Range value indicates 
        higher volatility, which may suggest larger price movements and 
        potentially greater trading opportunities or risks.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.

        Returns:
        numpy.ndarray: An array containing the True Range values.
        """
        matrix = self._extractor.extract_matrix(timeseries)
        high, low, close = matrix[:,2], matrix[:,3], matrix[:,4]
        a = high - low
        b = np.abs(high - np.roll(close, 1))
        c = np.abs(low  - np.roll(close, 1))
        tr = np.column_stack((a, b, c)).max(axis=1)
        tr[0] = a[0]
        return tr


    def _atr(self, timeseries, n=14):
        """
        Average True Range (ATR) indicator.

        ATR is a technical analysis indicator that measures market volatility by
        calculating the average of the True Range (TR) over a specified period.

        ATR is often used to assess the volatility of an asset and help traders
        identify potential trend reversals or continuation patterns. Higher ATR values
        indicate higher volatility, suggesting larger price movements and potentially
        greater trading opportunities or risks.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the ATR calculation. Default is 14.

        Returns:
        numpy.ndarray: An array containing the ATR values.
        """
        tr  = self._tr(timeseries)
        return self._wilders(tr, n)


    def _mfm(self, timeseries, smoothing=0, mode="sma"):
        """
        Money Flow Multiplier (MFM) indicator.
        
        MFM is a leading indicator that helps traders assess the relative strength 
        of buying or selling pressure in the market. It can provide insights into 
        potential changes in market sentiment before they are reflected in price 
        or volume data.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - smoothing (int): The number of periods to consider for smoothing the MFM
                           values. Default is 0 (no smoothing).
        - mode (str): The smoothing mode to use. Supported modes are "sma" (Simple
                      Moving Average), "ema" (Exponential Moving Average), etc.

        Returns:
        numpy.ndarray: An array containing the Money Flow Multiplier values.
        """
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        matrix = self._extractor.extract_matrix(timeseries)
        high, low, close = matrix[:,2], matrix[:,3], matrix[:,4]
        mfm = ((close - low) - (high - close)) / (high - low)
        if smoothing: mfm = getattr(self, f"_{mode}")(mfm, smoothing)
        return mfm


    def _er(self, timeseries, n=10, dimension="Close", smoothing=0, mode="ema"):
        """
        Kaufman Efficiency Ratio (ER) Indicator.

        Calculates the Efficiency Ratio (ER) of a given timeseries. ER is a technical indicator
        that measures the efficiency of a trend by comparing the price change over a certain
        number of periods to the volatility over the same period. It is useful for traders
        to assess the strength and sustainability of a trend.

        Parameters:
        - timeseries (pd.DataFrame or Stock): The input financial time series.
        - n (int): The number of periods to consider for the Efficiency Ratio calculation.
        - dimension (str): The column name representing the dimension in the timeseries.
        Typically, 'Close' is used for this parameter.
        - smoothing (int): The number of periods to consider for smoothing the ER
                           values. Default is 0 (no smoothing).
        - mode (str): The smoothing mode to use. Supported modes are "sma" (Simple
                      Moving Average), "ema" (Exponential Moving Average), etc.
        Returns:
        numpy.ndarray: An array containing the Efficiency Ratio values.
        """
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        absolute_differences = np.abs(np.diff(dimension))
        volatility = np.convolve(absolute_differences, np.ones(n-1), mode="valid")
        volatility = self._tools.resize(dimension, volatility)
        change = np.abs((dimension - np.roll(dimension, n - 1))[n - 1:])
        change = self._tools.resize(dimension, change)
        er = self._tools.divide(change, volatility)
        scaler = self._tools.sklearn_minmaxscaler(er, 0, 1)
        if smoothing: er = getattr(self, f"_{mode}")(er, smoothing)
        er = scaler.transform(er.reshape(-1, 1)).flatten()
        return er


    @classmethod
    def _update_fibonacci(cls, self, n):
        size    = len(cls._fibonacci)
        numbers = self._tools.fibonacci(n, size)
        if numbers is not None: cls._fibonacci = numbers
