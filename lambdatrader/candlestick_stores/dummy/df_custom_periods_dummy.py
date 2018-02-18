import time
from datetime import timedelta

import pandas

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.utilities.utils import seconds

pandas.set_option('display.max_rows', 30)
pandas.set_option('display.width', 200)

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

start_date = symbol_end_date - int(seconds(days=100))
df = store.get_df(symbol, start_date=start_date)

new_index = pandas.to_datetime(df.index, unit='s')

df = df.set_index(new_index)

print(df)

ohlc_dict = {
    'open':'first',
    'high':'max',
    'low':'min',
    'close': 'last',
    'base_volume': 'sum',
    'quote_volume': 'sum',
    'weighted_average': 'mean',
}

cols = df.columns.values.tolist()

df = df.resample('15T', closed='right', label='right').agg(ohlc_dict)
df = df[cols]

print()
print(df)


df = df.resample('D', closed='right', label='right').agg(ohlc_dict)
df = df[cols]

print()
print(df)
