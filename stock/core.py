from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import numpy as np


class Core:


    @property
    def timestamps(self):
        return self._data[:,0].astype("datetime64[m]")


    @property
    def close(self):
        return self._data[:,4]


    @property
    def Close(self):
        return self._data[:,4]


    @property
    def typical_price(self):
        return (self["high"] + self["low"] + self["close"]) / 3


    @property
    def empty(self):
        return False if self._data.shape[0] else True


    @property
    def shape(self):
        return self._data.shape


    @property
    def data(self):
        return self._data


    def __init__(self, symbol, timeframe, timescale=1, *args, **kwargs):
        self.db = InfluxDBClient()
        self.db.switch_database("nasdaq")
        self.symbol = symbol
        self.timeframe = timeframe
        self.timescale = timescale
        try: self._query()
        except Exception: self._data = np.array([])


    def __getitem__(self, row):
        if isinstance(row, str):
            if   row.lower() == "open":   return self._data[:,1]
            elif row.lower() == "high":   return self._data[:,2]
            elif row.lower() == "low" :   return self._data[:,3]
            elif row.lower() == "close":  return self._data[:,4]
            elif row.lower() == "volume": return self._data[:,5]
        else:
            return self._data[row]


    def _query(self):
        timestamp = datetime.now().date().strftime("%F")
        timestamp = np.datetime64(timestamp)
        timestamp = timestamp - np.timedelta64(self.timeframe, "D")
        timestamp = timestamp.astype(str) + "T9:30:00Z"
        q = f"""SELECT * FROM "ohlc" WHERE Symbol='{self.symbol}' 
        AND time >= '{timestamp}'"""
        r = self.db.query(q)
        data = r.raw["series"][0]["values"]
        df = pd.DataFrame(data, columns=r.raw["series"][0]["columns"])
        timestamps = [np.datetime64(t[:-1], "m") for t in df.time]
        timestamps = np.array(timestamps, dtype="datetime64[m]")
        timestamps = timestamps.astype(float)
        df.drop(columns=["time", "Close", "Symbol"], inplace=True)
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        matrix = df.to_numpy()
        matrix = np.column_stack((timestamps, matrix))
        self._data = matrix
