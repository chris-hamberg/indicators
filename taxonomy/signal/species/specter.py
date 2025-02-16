from indicators.base.core import CoreIndicators
import numpy as np
import pywt


class WaveSpecter(CoreIndicators):
    """
    Computes a wavelet spectrogram for the given volume time series.

    The wavelet spectrogram provides a representation of how the frequency 
    content of the time series changes over time, which can be valuable for 
    identifying patterns or trends in trading volume.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - dimension (str): The column name or key in the time series.
    - wavelet (str): The wavelet function to use for the wavelet transform.
    - cmap (str): The color map to use for the spectrogram plot.
    - extent (int, optional): The extent of the spectrogram plot.

    Returns:
    - numpy.ndarray: The wavelet spectrogram matrix.
    - dict: Additional keyword arguments for plotting, including 'aspect', 
    'cmap', 'extent', and 'interpolation'.
    """
    def __init__(self, aspect="auto", interpolation="bilinear"):
        super().__init__()
        self.interpolation = interpolation
        self.aspect        = aspect


    def __call__(self, timeseries, dimension, wavelet, cmap, extent):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        matrix, freqs = self.matrix_maker(dimension, wavelet)
        minima = self._tools.nanmin_ignore_nan(freqs)
        maxima = self._tools.nanmax_ignore_nan(freqs)
        if not extent: extent = dimension.shape[0]
        extent = [0, extent, minima, maxima]
        kwargs = {"aspect": self.aspect, "cmap": cmap, "extent": extent,
                "interpolation": self.interpolation}
        return matrix, kwargs


    def matrix_maker(self, dimension, wavelet):
        widths = np.arange(1, 11)
        matrix, freqs = pywt.cwt(dimension, widths, wavelet)
        matrix = np.abs(matrix)
        return matrix, freqs
