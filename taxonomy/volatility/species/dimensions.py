from indicators.base.core import CoreIndicators
import numpy as np


class Dimensions(CoreIndicators):
    """
    Dimensions indicator.

    The Dimensions indicator calculates the fractal dimensions of a time series 
    based on different modes of analysis (flat, expanding, or rolling). Fractal 
    dimensions can provide insights into the self-similarity and complexity of 
    a time series.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
      should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the fractal analysis.
    - dimension (str): The column name representing the dimension in the 
      timeseries. This could be 'close' for price data or 'volume' for volume data.
    - mode (str): The mode of analysis to use. Options are 'flat', 'expanding', or 'rolling'.

    Returns:
    numpy.ndarray: An array containing the calculated fractal dimensions.
    """
    modes = {

            "flat"      : {
                            "X2"   : lambda n, X1: [X1.shape[0] - 1], 
                            "start": lambda i, n: 0,
                            "end"  : lambda i: None},

            "expanding" : {
                            "X2"   : lambda n, X1: range(1, X1.shape[0]),
                            "start": lambda i, n: 0,
                            "end"  : lambda i: i},

            "rolling"   : {
                            "X2"   : lambda n, X1: range(n, X1.shape[0]), 
                            "start": lambda i, n: i-n+1,
                            "end"  : lambda i: i+1}}


    def __init__(self):
        super().__init__()
        self.error = f"Mode must be one of: {list(Dimensions.modes.keys())}"


    def __call__(self, timeseries, n, dimension, mode):
        if not mode in Dimensions.modes.keys(): raise ValueError(self.error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        space     = Dimensions.modes[mode]["X2"](n, dimension)
        start     = Dimensions.modes[mode]["start"]
        end       = Dimensions.modes[mode]["end"]
        dimension = [self.fractals(dimension[start(i,n):end(i)]) for i in space]
        pad       = np.array(n * [np.nan, np.nan]).reshape(-1, 2)
        dimension = np.vstack((pad, dimension))
        return (dimension[:,0], dimension[:,1]) if mode != "flat" else dimension[-1]


    def fractals(self, subspace):
        subspace           = (subspace[:,None] - subspace)**2
        pairwise_distances = np.sqrt(np.sum(subspace, axis=1))
        avg_distance       = np.mean(pairwise_distances)
        box_sizes          = np.logspace(0.1, 1.5, 20).astype(int)
        box_sizes          = np.maximum(box_sizes, 1)
        counts             = pairwise_distances < box_sizes[:,None]
        counts             = np.sum(counts, axis=1)
        counts             = np.maximum(counts, 1)
        coeffs             = np.polyfit(np.log(box_sizes), np.log(counts), 1)
        dimensions         = -coeffs[0]
        return np.array((dimensions, avg_distance))
