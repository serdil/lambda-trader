from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.factories import FeatureSets as fsf
from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.dmatrix_save_load_util \
    import (
    save_close_dmatrix, save_max_dmatrix,
)

all_symbols = set(SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs())


# symbols = sorted(list(all_symbols))
# num_candles = 48
# day_offset = 12
# days = 200
# val_ratio = 0.8
# test_ratio = 0.9
# feature_set = fsf.get_small()

# symbols = sorted(list(all_symbols))
# num_candles = 48
# day_offset = 12
# days = 400
# val_ratio = 0.90
# test_ratio = 0.95
# feature_set = fsf.get_small()

# symbols = sorted(list(all_symbols))
# num_candles = 48
# day_offset = 12
# days = 200
# val_ratio = 0.8
# test_ratio = 0.9
# feature_set = fsf.get_all_periods_last_five_ohlcv()

# symbols = sorted(list(all_symbols))
# num_candles = 48
# day_offset = 12
# days = 200
# val_ratio = 0.8
# test_ratio = 0.9
# feature_set = fsf.get_all_periods_last_ten_ohlcv()

# symbols = sorted(list(all_symbols))
# num_candles = 48
# day_offset = 12
# days = 7
# val_ratio = 0.6
# test_ratio = 0.8
# feature_set = fsf.get_all_periods_last_ten_ohlcv()

symbols = sorted(list(all_symbols))
num_candles = 1
day_offset = 12
days = 200
val_ratio = 0.8
test_ratio = 0.9
feature_set = fsf.get_all_periods_last_ten_ohlcv_now_delta()

valr = int(val_ratio * 100)
testr = int(test_ratio * 100)
save_close_dmatrix(num_candles=num_candles, candle_period=M5, feature_set=feature_set,
                   num_days=days, days_offset=day_offset, symbols=symbols, valr=valr, testr=testr)
save_max_dmatrix(num_candles=num_candles, candle_period=M5, feature_set=feature_set,
                 num_days=days, days_offset=day_offset, symbols=symbols, valr=valr, testr=testr)
