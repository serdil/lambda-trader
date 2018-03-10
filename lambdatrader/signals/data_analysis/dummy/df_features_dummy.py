import pandas

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5, H, H4, D
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.constants import OHLCV_CLOSE, OHLCV_OPEN
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVNowCloseDelta, IndicatorValue, RSIValue, BBandsNowCloseDelta, MACDValue, DummyFeature,
    RandomFeature, OHLCVSelfCloseDelta,
)
from lambdatrader.utilities.utils import seconds

pandas.set_option('display.max_rows', 20)
pandas.set_option('display.width', 200)

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

symbol_end_date = store.get_pair_period_newest_date(symbol)

start_date = symbol_end_date - int(seconds(days=10))
dfs = store.get_agg_period_dfs(symbol, start_date=start_date, periods=[M5, H, H4, D])

close_delta_feature = OHLCVNowCloseDelta(OHLCV_CLOSE, offset=1, period=M5)

print(dfs[M5])

# print('close_delta')
# print(close_delta_feature.compute(dfs))
#
# indicator_value_feature = IndicatorValue(IndicatorEnum.SMA, [14], offset=0, period=M5,
#                                          longest_timeperiod=14)
#
# print('sma')
# print(indicator_value_feature.compute(dfs))
#
# rsi_feature = RSIValue()
# bbands_feature = BBandsCloseDelta()
# macd_feature = MACDValue()
#
#
# print('rsi')
# print(rsi_feature.compute(dfs))
#
# print('bbands')
# print(bbands_feature.compute(dfs))
#
# print('macd')
# print(macd_feature.compute(dfs))
#
# bbands_h = BBandsCloseDelta(period=H)
# print('bbands H')
# print(bbands_h.compute(dfs))
#
#
# dummy = DummyFeature()
# print('dummy')
# print(dummy.compute(dfs))
#
#
# random = RandomFeature()
# print('random')
# print(random.compute(dfs))
#

# close_self_close_delta = OHLCVSelfCloseDelta(OHLCV_CLOSE, offset=1, period=M5)
# print('close self close delta')
# print(close_self_close_delta.compute(dfs))

open_self_close_delta = OHLCVSelfCloseDelta(OHLCV_OPEN, offset=0, period=M5)
print('open self close delta')
print(open_self_close_delta.compute(dfs))
