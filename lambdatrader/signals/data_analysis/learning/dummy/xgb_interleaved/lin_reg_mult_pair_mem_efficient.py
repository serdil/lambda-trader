import numpy as np

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory as fsf
from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.utils import (
    get_test_X_ys, train_close_from_saved, train_max_from_saved, train_close, train_max,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output

all_symbols = set(SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs())

# symbols = ['BTC_ETH']
symbols = sorted(list(all_symbols))

num_candles = 48

day_offset = 12
days = 200

val_ratio = 0.8
test_ratio = 0.9

feature_set = fsf.get_all_periods_last_ten_ohlcv()

use_saved = True
# use_saved = False

params = {
    'silent': 1,
    'booster': 'gblinear',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.1,
    'gamma': 0,
    'max_depth': 3,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'tree_method': 'exact',
    'sketch_eps': 0.03,
    'scale_pos_weight': 1,
    'refresh_leaf': 1,
    'process_type': 'default',
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'max_bin': 256,

    # 'lambda': 1,
    # 'alpha': 0,
    # 'updater': 'grow_colmaker,prune',

    # 'sample_type': 'uniform',
    # 'normalize_type': 'tree',
    # 'rate_drop': 0.00,
    # 'one_drop': 0,
    # 'skip_drop': 0.00,

    'reg_lambda': 0,
    'reg_alpha': 0,
    # 'updater': 'shotgun'
}

close_params = params.copy()

max_params = params.copy()
max_params.update({
    'eta': 0.2
})

num_rounds = 100
early_stopping_rounds = 10

common_args = {
    'day_offset': day_offset,
    'days': days,
    'early_stopping_rounds': early_stopping_rounds,
    'feature_set': feature_set,
    'num_candles': num_candles,
    'num_rounds': num_rounds,
    'symbols': symbols,
    'test_ratio': test_ratio,
    'val_ratio': val_ratio
}

if use_saved:
    pred_close = train_close_from_saved(params=close_params, **common_args)
    pred_max = train_max_from_saved(params=max_params, **common_args)
else:
    pred_close = train_close(params=close_params, **common_args)
    pred_max = train_max(params=max_params, **common_args)

print()
print('++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++')
print()

pred_min = np.zeros(len(pred_close))
y_min_test = np.zeros(len(pred_close))

X_test, y_close_test, y_max_test = get_test_X_ys(**common_args)

pred_real_close = list(zip(pred_close, y_close_test))
pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))

analyze_output(pred_real_close=pred_real_close,
               pred_real_max=pred_real_max,
               pred_real_min=pred_real_min)
