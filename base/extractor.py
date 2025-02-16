from pandas.core.frame import DataFrame
from indicators.stock import Stock
import numpy as np


class Extractor:


    _dimensions  = ["Open", "High", "Low", "Close", "Volume"]
    _ldimensions = ["open", "high", "low", "close", "volume"]


    def __init__(self):
        self.dimension_error = "Datatype does not meet expectation."
        self.matrix_error    = "Datatype must be 2-dimensional."


    def extract_dimension(self, timeseries, dimension="Close"):

        if isinstance(timeseries, DataFrame):
            try: timeseries = timeseries[dimension].values
            except KeyError:
                timeseries = timeseries[dimension.lower()].values
        
        elif isinstance(timeseries, Stock):
            timeseries = timeseries[dimension]
        
        elif isinstance(timeseries, np.ndarray):
            if timeseries.ndim == 2 and timeseries.shape[1] == 6:
                dimension = Extractor._dimensions.index(dimension) + 1
                timeseries = timeseries[:,dimension]
            elif timeseries.ndim != 1:
                raise TypeError(self.dimension_error)
        
        return timeseries


    def extract_matrix(self, timeseries):
        
        if isinstance(timeseries, DataFrame):
            dimension  = np.full(timeseries.shape[0], np.nan)
            try: timeseries = timeseries[Extractor._dimensions].to_numpy()
            except KeyError:
                timeseries = timeseries[Extractor._ldimensions].to_numpy()
            timeseries = np.column_stack((dimension, timeseries))
        
        elif isinstance(timeseries, Stock):
            timeseries = timeseries.data
        
        elif isinstance(timeseries, np.ndarray):
            if timeseries.ndim != 2: 
                raise TypeError(self.matrix_error)
        
        return timeseries


    def convert_to_stock(self, timeseries):

        if isinstance(timeseries, DataFrame):
            timeseries = Stock(dataframe=timeseries)

        elif isinstance(timeseries, np.ndarray):
            if timeseries.ndim == 6:
                timeseries = Stock(matrix=timeseries)
            elif timeseries.ndim == 1:
                timeseries = Stock(series=timeseries)

        return timeseries


if __name__ == "__main__":
    ex     = Extractor()
    stock  = Stock("AAPL", remote=False)
    matrix = stock.data
    ex.extract_dimension(matrix)
