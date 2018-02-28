from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory as fsf
from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.dmatrix_save_load_util \
    import (
    save_close_dmatrix, load_close_dmatrix,
)

all_symbols = set(SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs())


symbols = sorted(list(all_symbols))
num_candles = 48
day_offset = 12
days = 5
val_ratio = 0.8
test_ratio = 0.9
feature_set = fsf.get_small()


valr = int(val_ratio * 10)
testr = int(test_ratio * 10)
save_close_dmatrix(num_candles=num_candles, candle_period=M5, feature_set=feature_set,
                   num_days=days, days_offset=day_offset, symbols=symbols, valr=valr, testr=testr)

loadeds = load_close_dmatrix(num_candles=num_candles, candle_period=M5, feature_set=feature_set,
                             num_days=days, days_offset=day_offset, symbols=symbols,
                             valr=valr, testr=testr)

print(loadeds)
