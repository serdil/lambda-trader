import pandas

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.constants import OHLCV_CLOSE
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVCloseDelta, IndicatorValue, RSIValue, BBandsCloseDelta, MACDValue,
)
from lambdatrader.utilities.utils import seconds

pandas.set_option('display.max_rows', 20)
pandas.set_option('display.width', 200)

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

start_date = symbol_end_date - int(seconds(days=1))
df = store.get_df(symbol, start_date=start_date)

close_delta_feature = OHLCVCloseDelta(OHLCV_CLOSE, offset=1, period=M5)

print(df)

print('close_delta')
print(close_delta_feature.compute(df))

indicator_value_feature = IndicatorValue(IndicatorEnum.SMA, [14], offset=0, period=M5)

print('sma')
print(indicator_value_feature.compute(df))

rsi_feature = RSIValue()
bbands_feature = BBandsCloseDelta()
macd_feature = MACDValue()

print('rsi')
print(rsi_feature.compute(df))

print('bbands')
print(bbands_feature.compute(df))

print('macd')
print(macd_feature.compute(df))
