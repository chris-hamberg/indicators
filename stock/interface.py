from stock.core import Core
import pandas as pd
import numpy as np


class Days(list):


    def __init__(self):
        super().__init__()
        self._index = list()


    def append(self, day):
        super().append(day)
        self._index.append(day.date)


    def lookup(self, timestamp):
        date = np.datetime64(timestamp, "D")
        try: index = self._index.index(date)
        except ValueError: 
            raise IndexError("Timestamp date not found")
        else: return self[index]


class Stock(Core):


    def __init__(self, symbol, timeframe, timescale=1, *args, **kwargs):
        super().__init__(symbol, timeframe, timescale, *args, **kwargs)
        self.days = Days()
        self._parse()


    def _parse(self):
        try: df, closes = self._closes()
        except IndexError: return 0
        for i in range(len(closes) - 1):
            start = closes[i][0] < df.timestamp
            end   = df.timestamp < closes[i+1][0]
            frame = df.loc[(start) & (end)]
            prev  = closes[i][-1]
            day   = Day(prev, frame)
            self.days.append(day)


    def _closes(self):
        timestamps = pd.to_datetime(self.timestamps)
        closes     = self.close
        df         = pd.DataFrame({"timestamp": timestamps, "close": closes})
        df["date"] = df.timestamp.dt.floor("D")
        closes     = df.groupby("date").last().reset_index()
        closes     = list(zip(closes["timestamp"], closes["close"]))
        return df, closes


class Day:


    @property
    def data(self):
        return self._data


    @property
    def prev(self):
        return self._prev


    @property
    def timestamps(self):
        return self._data[:,0].astype("datetime64[m]")


    @property
    def close(self):
        return self._data[:,1]


    @property
    def pct(self):
        return self._data[:,2]


    @property
    def date(self):
        return self._date


    def __init__(self, prev, frame):
        try: 
            date = frame.timestamp.iloc[-1].date()
            date = np.datetime64(date, "D")
        except IndexError:
            date = None
        timestamps = frame.timestamp.values.astype("datetime64[m]")
        timestamps = timestamps.astype(float)
        pct        = (prev - frame.close) / frame.close
        close      = frame["close"]
        self._prev = prev
        self._date = date
        self._data = np.column_stack((timestamps, close, pct))


    def lookup(self, timestamp):
        if isinstance(timestamp, float):
            timestamp = np.datetime64(timestamp, "m")
        elif not isinstance(timestamp, np.datetime64):
            raise ValueError("timestamp must be np.datetime64 or float")
        mask = self.timestamps <= timestamp
        if np.any(mask):
            return self._data[mask][-1, 2]
        return None
