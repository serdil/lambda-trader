import pandas

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5, H, H4, D
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_values import (
    MaxReturn, CloseReturn, MinReturn, CloseAvgReturn,
)
from lambdatrader.utilities.utils import seconds

pandas.set_option('display.max_rows', 20)
pandas.set_option('display.width', 200)

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

start_date = symbol_end_date - int(seconds(days=10))
dfs = store.get_agg_period_dfs(symbol, start_date=start_date, periods=[M5, H, H4, D])

print(dfs[M5])

mrv = MaxReturn(n_candles=3)
crv = CloseReturn(n_candles=3)
minrv = MinReturn(n_candles=3)
cavgrv = CloseAvgReturn(n_candles=3)

print()
print(mrv.compute(dfs))

print()
print(crv.compute(dfs))

print()
print(minrv.compute(dfs))

print()
print(cavgrv.compute(dfs))
