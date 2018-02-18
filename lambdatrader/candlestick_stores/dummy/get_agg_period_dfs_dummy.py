import time

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M15, D, H4, H, M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.utilities.utils import seconds

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

small_start_date = symbol_end_date - int(seconds(days=1))
df_small = store.get_df(symbol, start_date=small_start_date)

print(df_small)

start_time = time.time()
df_large = store.get_df(symbol)
duration = time.time() - start_time

print(df_large.describe())
print()
print('duration:', duration)


start_time = time.time()
df_large_m15 = store.get_agg_period_df(symbol, period=M15)
duration = time.time() - start_time

print(df_large_m15)
print('m15 duration:', duration)

start_time = time.time()
df_large_d = store.get_agg_period_df(symbol, period=D)
duration = time.time() - start_time

print(df_large_d)
print('D duration:', duration)


start_time = time.time()
dfs = store.get_agg_period_dfs(symbol, periods=[M5, M15, H, H4, D])
duration = time.time() - start_time

print('M5, M15, H, H4, D duration:', duration)
