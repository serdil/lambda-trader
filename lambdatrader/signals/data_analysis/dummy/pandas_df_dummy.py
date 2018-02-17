import pandas as pd

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.utilities.utils import seconds

pd.set_option('display.max_rows', 10)

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

small_start_date = symbol_end_date - int(seconds(days=1))
df_small = store.get_df(symbol, start_date=small_start_date)


# print(df_small)

close = df_small['close']

print(close)

print()
print(close.diff(1))


print()
print(close - close.shift(1))

