# get max prices, min prices and close prices for 4 hours
# train model to predict max price in 4 hours
# train model to predict min price in 4 hours
# train model to predict close price in 4 hours
# determine trading gains for different
# (max_price_threshold, min_price_threshold, close_price_threshold) pairs.

from datetime import datetime

import xgboost as xgb

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import CandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set, get_dummy_feature_func_set,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output
from lambdatrader.signals.data_analysis.values import (
    make_cont_max_price_in_future, make_cont_close_price_in_future, make_cont_min_price_in_future,
)
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

day_offset = 7

# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*500)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*365)
dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*250)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*200)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*120)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*90)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*30)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24)
# dataset_start_date = latest_market_date - seconds(days=day_offset, minutes=30)

dataset_end_date = latest_market_date - seconds(days=day_offset)

dataset_len = dataset_end_date - dataset_start_date

print('start_date: {} end_date: {}'.format(datetime.utcfromtimestamp(dataset_start_date),
                                         datetime.utcfromtimestamp(dataset_end_date)))
print()


dataset_symbol = 'BTC_SC'

print('symbol:', dataset_symbol)

dummy_feature_functions = list(get_dummy_feature_func_set())
dummy_feature_functions_name = 'dummy'

feature_functions = list(get_small_feature_func_set())
feature_funcs_name = 'small'

num_candles = 48

max_price_value_func = make_cont_max_price_in_future(num_candles=num_candles, candle_period=M5)
max_price_value_func_name = 'cont_max_price_{}'.format(num_candles)

min_price_value_func = make_cont_min_price_in_future(num_candles=num_candles, candle_period=M5)
min_price_value_func_name = 'cont_min_price_{}'.format(num_candles)

close_price_value_func = make_cont_close_price_in_future(num_candles=num_candles, candle_period=M5)
close_price_value_func_name = 'cont_close_price_{}'.format(num_candles)

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=feature_functions,
                                           value_function=max_price_value_func,
                                           cache_and_get_cached=True,
                                           feature_functions_key=feature_funcs_name,
                                           value_function_key=max_price_value_func_name)

max_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                           pair=dataset_symbol,
                                                           start_date=dataset_start_date,
                                                           end_date=dataset_end_date,
                                                           feature_functions=dummy_feature_functions,
                                                           value_function=max_price_value_func,
                                                           cache_and_get_cached=True,
                                                           feature_functions_key=dummy_feature_functions_name,
                                                           value_function_key=max_price_value_func_name)

min_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                           pair=dataset_symbol,
                                                           start_date=dataset_start_date,
                                                           end_date=dataset_end_date,
                                                           feature_functions=dummy_feature_functions,
                                                           value_function=min_price_value_func,
                                                           cache_and_get_cached=True,
                                                           feature_functions_key=dummy_feature_functions_name,
                                                           value_function_key=min_price_value_func_name)

close_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                             pair=dataset_symbol,
                                                             start_date=dataset_start_date,
                                                             end_date=dataset_end_date,
                                                             feature_functions=dummy_feature_functions,
                                                             value_function=close_price_value_func,
                                                             cache_and_get_cached=True,
                                                             feature_functions_key=dummy_feature_functions_name,
                                                             value_function_key=close_price_value_func_name)


feature_names = dataset.get_first_feature_names()
X = dataset.get_numpy_feature_matrix()
y_max = max_price_value_dataset.get_numpy_value_array()
y_min = min_price_value_dataset.get_numpy_value_array()
y_close = close_price_value_dataset.get_numpy_value_array()

print('created/loaded dataset\n')


def days_to_candles(num_days):
    return seconds(days=num_days) // M5.seconds()


train_val_days = 200
train_ratio = 0.9

train_days = int(train_val_days * train_ratio)
validation_days = train_val_days - train_days

retrain_interval_days = validation_days

train_len = days_to_candles(train_days)
val_len = days_to_candles(validation_days)
retrain_len = days_to_candles(retrain_interval_days)

gap = num_candles

n_samples = len(X)

window_size = train_len + gap + val_len + gap + retrain_len

nontest_len = window_size - retrain_len
test_end = n_samples - ((n_samples - nontest_len) % retrain_len)

total_test_days = (test_end - nontest_len) // days_to_candles(1)
num_training_rounds = total_test_days // retrain_interval_days
print('total test days:', total_test_days)
print('num training rounds:', num_training_rounds)

# print('window size', window_size)
# print('n_samples', n_samples)
# print('nontest len', nontest_len)
# print('test end', test_end)

pred_max = []  # TODO convert to numpy array
pred_min = []
pred_close = []

y_test_max = y_max[nontest_len:test_end]
y_test_min = y_min[nontest_len:test_end]
y_test_close = y_close[nontest_len:test_end]

for start_ind in range(0, n_samples-window_size+1, retrain_len):
    print('training round')

    train_start = start_ind
    train_end = start_ind + train_len

    val_start = train_end + gap
    val_end = val_start + val_len

    test_start = val_end + gap
    test_end = test_start + retrain_len

    # print('train start - val end:', train_start, val_end)
    # print('test start-end', test_start, test_end)

    assert test_end - train_start == window_size

    X_train = X[train_start:train_end]
    y_max_train = y_max[train_start:train_end]
    y_min_train = y_min[train_start:train_end]
    y_close_train = y_close[train_start:train_end]

    X_val = X[val_start:val_end]
    y_max_val = y_max[val_start:val_end]
    y_min_val = y_min[val_start:val_end]
    y_close_val = y_close[val_start:val_end]

    X_test = X[test_start:test_end]
    y_max_test = y_max[test_start:test_end]
    y_min_test = y_min[test_start:test_end]
    y_close_test = y_close[test_start:test_end]

    dtrain_max = xgb.DMatrix(X_train, label=y_max_train, feature_names=feature_names)
    dval_max = xgb.DMatrix(X_val, label=y_max_val, feature_names=feature_names)
    dtest_max = xgb.DMatrix(X_test, label=y_max_test, feature_names=feature_names)

    dtrain_min = xgb.DMatrix(X_train, label=y_min_train, feature_names=feature_names)
    dval_min = xgb.DMatrix(X_val, label=y_min_val, feature_names=feature_names)
    dtest_min = xgb.DMatrix(X_test, label=y_min_test, feature_names=feature_names)

    dtrain_close = xgb.DMatrix(X_train, label=y_close_train, feature_names=feature_names)
    dval_close = xgb.DMatrix(X_val, label=y_close_val, feature_names=feature_names)
    dtest_close = xgb.DMatrix(X_test, label=y_close_test, feature_names=feature_names)

    params = {
        'silent': 1, 'booster': 'gblinear',

        'objective': 'reg:linear', 'base_score': 0, 'eval_metric': 'rmse',

        'eta': 0.01, 'gamma': 0, 'max_depth': 3, 'min_child_weight': 2, 'max_delta_step': 0,
        'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'lambda': 0, 'alpha': 0,
        'tree_method': 'auto', 'sketch_eps': 0.03, 'scale_pos_weight': 1,
        'updater': 'grow_colmaker,prune', 'refresh_leaf': 1, 'process_type': 'default',
        'grow_policy': 'depthwise', 'max_leaves': 0, 'max_bin': 256,

        'sample_type': 'weighted', 'rate_drop': 0.01,
    }

    watchlist_max = [(dtrain_max, 'train_max'), (dtest_max, 'test_max'), (dval_max, 'val_max')]
    watchlist_min = [(dtrain_min, 'train_min'), (dtest_min, 'test_min'), (dval_min, 'val_min')]
    watchlist_close = [(dtrain_close, 'train_close'), (dtest_close, 'test_close'),
                       (dval_close, 'val_close')]

    num_round = 10000
    early_stopping_rounds = 100

    bst_max = xgb.train(params=params, dtrain=dtrain_max, num_boost_round=num_round,
                        evals=watchlist_max, early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=False)

    bst_min = xgb.train(params=params, dtrain=dtrain_min, num_boost_round=num_round,
                        evals=watchlist_min, early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=False)

    bst_close = xgb.train(params=params, dtrain=dtrain_close, num_boost_round=num_round,
                          evals=watchlist_close, early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=False)

    p_pred_max = bst_max.predict(dtest_max)
    p_pred_min = bst_min.predict(dtest_min)
    p_pred_close = bst_close.predict(dtest_close)

    pred_max.extend(p_pred_max)
    pred_min.extend(p_pred_min)
    pred_close.extend(p_pred_close)


assert len(pred_max) == len(y_test_max)
assert len(pred_min) == len(y_test_min)
assert len(pred_close) == len(y_test_close)

pred_real_max = list(zip(pred_max, y_test_max))
pred_real_min = list(zip(pred_min, y_test_min))
pred_real_close = list(zip(pred_close, y_test_close))


analyze_output(pred_real_max, pred_real_min, pred_real_close)
