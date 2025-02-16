from indicators.taxonomy.crayon.genus.bullish.whitebody import WhiteBody
import numpy as np


class BigGreenCrayon:


    def __init__(self):
        self.white_body = WhiteBody()


    def __call__(self, metacrayon):

        # Threshold
        threshold = metacrayon.average_body_size + 2 * metacrayon.std_body_size

        # Open is less than close
        color = self.white_body(metacrayon)

        # Large body
        body_size = threshold <= metacrayon.body_size

        # Shape
        divisor = metacrayon.range_size
        delta   = (metacrayon.range_size != 0) & ~(np.isnan(metacrayon.range_size))

        # Open is near the high
        dividend = metacrayon.high - metacrayon.open
        top      = np.zeros(metacrayon.open.shape[0])
        top      = np.divide(dividend, divisor, out=top, where=delta)
        top      = np.where((top <= 0.1) & (top != 0), True, False)

        # Close is near the low
        dividend = metacrayon.close - metacrayon.low
        bottom   = np.zeros(metacrayon.close.shape[0])
        bottom   = np.divide(dividend, divisor, out=bottom, where=delta)
        bottom = np.where((bottom <= 0.1) & (bottom != 0), True, False)

        return np.where((color) & (body_size) & (top) & (bottom), 1, 0)
