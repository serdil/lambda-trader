import pandas as pd


from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.utilities.utils import seconds

pd.set_option('display.max_rows', 10)

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

small_start_date = symbol_end_date - int(seconds(days=1))
df_small = store.get_df(symbol, start_date=small_start_date)

sma = IndicatorEnum.SMA.function()
macd = IndicatorEnum.MACD.function()
bbands = IndicatorEnum.BBANDS.function()


print(df_small)

print()
print(sma(df_small, 3))


macd_out = macd(df_small)

print()
print(macd_out)

print()
print(df_small['close'])

bbands_out = bbands(df_small)

print()
print(bbands_out)

print()
print(bbands_out.rsub(df_small['close'], axis=0))

