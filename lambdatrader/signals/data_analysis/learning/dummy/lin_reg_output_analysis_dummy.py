# get max prices, min prices and close prices for 4 hours
# train model to predict max price in 4 hours
# train model to predict min price in 4 hours
# train model to predict close price in 4 hours
# determine trading gains for different
# (max_price_threshold, min_price_threshold, close_price_threshold) pairs.

import math

from operator import itemgetter

import numpy as np
import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set, get_dummy_feature_func_set,
    get_small_feature_func_set_with_indicators,
)
from lambdatrader.signals.data_analysis.values import (
    make_cont_max_price_in_future, make_cont_close_price_in_future, make_cont_min_price_in_future,
)
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

day_offset = 60

# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*500)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*365)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*200)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*120)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*90)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*30)
dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24)
# dataset_start_date = latest_market_date - seconds(days=day_offset, minutes=30)

dataset_end_date = latest_market_date - seconds(days=day_offset)

dataset_len = dataset_end_date - dataset_start_date


dataset_symbol = 'BTC_ETC'

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

train_ratio = 0.6
validation_ratio = 0.8
gap = num_candles

n_samples = len(X)

validation_split_ind = int(train_ratio * n_samples)
test_split_ind = int(validation_ratio * n_samples)

X_train = X[:validation_split_ind - gap]
y_max_train = y_max[:validation_split_ind - gap]
y_min_train = y_min[:validation_split_ind - gap]
y_close_train = y_close[:validation_split_ind - gap]


X_val = X[validation_split_ind:test_split_ind]
y_max_val = y_max[validation_split_ind:test_split_ind]
y_min_val = y_min[validation_split_ind:test_split_ind]
y_close_val = y_close[validation_split_ind:test_split_ind]


X_test = X[test_split_ind:]
y_max_test = y_max[test_split_ind:]
y_min_test = y_min[test_split_ind:]
y_close_test = y_close[test_split_ind:]


# print(X_test)
# print(y_max_test)
# print(y_min_test)
# print(y_close_test)

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
    'silent': 1,
    'booster': 'gblinear',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.01,
    'gamma': 0,
    'max_depth': 3,
    'min_child_weight': 2,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'lambda': 0,
    'alpha': 0,
    'tree_method': 'auto',
    'sketch_eps': 0.03,
    'scale_pos_weight': 1,
    'updater': 'grow_colmaker,prune',
    'refresh_leaf': 1,
    'process_type': 'default',
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'max_bin': 256,

    'sample_type': 'weighted',
    'rate_drop': 0.01,
}

watchlist_max = [(dtrain_max, 'train_max'), (dtest_max, 'test_max'), (dval_max, 'val_max')]
watchlist_min = [(dtrain_min, 'train_min'), (dtest_min, 'test_min'), (dval_min, 'val_min')]
watchlist_close = [(dtrain_close, 'train_close'),
                   (dtest_close, 'test_close'), (dval_close, 'val_close')]


num_round = 10000
early_stopping_rounds = 100

bst_max = xgb.train(params=params,
                    dtrain=dtrain_max,
                    num_boost_round=num_round,
                    evals=watchlist_max,
                    early_stopping_rounds=early_stopping_rounds)

bst_min = xgb.train(params=params,
                    dtrain=dtrain_min,
                    num_boost_round=num_round,
                    evals=watchlist_min,
                    early_stopping_rounds=early_stopping_rounds)

bst_close = xgb.train(params=params,
                      dtrain=dtrain_close,
                      num_boost_round=num_round,
                      evals=watchlist_close,
                      early_stopping_rounds=early_stopping_rounds)


max_best_ntree_limit = bst_max.best_ntree_limit
min_best_ntree_limit = bst_min.best_ntree_limit
close_best_ntree_limit = bst_close.best_ntree_limit


feature_importances_max = bst_max.get_fscore()
feature_importances_min = bst_min.get_fscore()
feature_importances_close = bst_close.get_fscore()

print()
print('feature importances max:')
for f_name, imp in list(reversed(sorted(feature_importances_max.items(), key=itemgetter(1))))[:10]:
    print(f_name, ':', imp)

print()
print('feature importances min:')
for f_name, imp in list(reversed(sorted(feature_importances_min.items(), key=itemgetter(1))))[:10]:
    print(f_name, ':', imp)

print()
print('feature importances close:')
for f_name, imp in list(reversed(sorted(feature_importances_close.items(), key=itemgetter(1))))[:10]:
    print(f_name, ':', imp)


def max_pred_key(a):
    return a[0][0]


def min_pred_key(a):
    return a[1][0]


def close_pred_key(a):
    return a[2][0]


def max_real_key(a):
    return a[0][1]


def min_real_key(a):
    return a[1][1]


def close_real_key(a):
    return a[2][1]


tp_level = 0.9


def tp_at_close_pred_profit(a):
    return close_pred_key(a) * tp_level \
        if one_if_tp_at_close_pred_hit(a) == 1 \
        else close_real_key(a)


def one_if_tp_at_close_pred_hit(a):
    return 1 \
        if close_pred_key(a) > 0 and max_real_key(a) >= close_pred_key(a) * tp_level \
        else 0


def get_tp_at_close_pred_thr_profit(thr):
    def tp_at_close_pred_thr_profit(a):
        return thr * tp_level \
            if get_one_if_tp_at_close_pred_thr_hit(thr)(a) == 1 \
            else close_real_key(a)
    return tp_at_close_pred_thr_profit


def get_one_if_tp_at_close_pred_thr_hit(thr):
    def one_if_tp_at_close_pred_thr_hit(a):
        return 1 \
            if thr > 0 and close_real_key(a) >= thr * tp_level \
            else 0
    return one_if_tp_at_close_pred_thr_hit


def tp_at_max_pred_profit(a):
    return max_pred_key(a) * tp_level \
        if one_if_tp_at_max_pred_hit(a) == 1 \
        else close_real_key(a)


def one_if_tp_at_max_pred_hit(a):
    return 1 \
        if max_pred_key(a) > 0 and max_real_key(a) >= max_pred_key(a) * tp_level \
        else 0


def get_tp_at_max_pred_thr_profit(thr):
    def tp_at_max_pred_thr_profit(a):
        return thr * tp_level \
            if get_one_if_tp_at_max_pred_thr_hit(thr)(a) == 1 \
            else close_real_key(a)
    return tp_at_max_pred_thr_profit


def get_one_if_tp_at_max_pred_thr_hit(thr):
    def one_if_tp_at_max_pred_thr_hit(a):
        return 1 \
            if thr > 0 and max_real_key(a) >= thr * tp_level \
            else 0
    return one_if_tp_at_max_pred_thr_hit


def one_if_max_pred_hit(a):
    return 1 if max_real_key(a) >= max_pred_key(a) else 0


def one_if_min_pred_hit(a):
    return 1 if min_real_key(a) >= min_pred_key(a) else 0


def one_if_close_pred_hit(a):
    return 1 if close_real_key(a) >= close_pred_key(a) else 0


def analyze_output(pred_real_max, pred_real_min, pred_real_close):
    pred_real_max_min_close = list(zip(pred_real_max, pred_real_min, pred_real_close))

    profits = [tp_at_close_pred_profit(a) for a in pred_real_max_min_close]

    pred_real_max_min_close_profit = list(zip(pred_real_max_min_close, profits))

    minimum_max_pred = max_pred_key(min(pred_real_max_min_close, key=max_pred_key))
    maximum_max_pred = max_pred_key(max(pred_real_max_min_close, key=max_pred_key))

    minimum_min_pred = min_pred_key(min(pred_real_max_min_close, key=min_pred_key))
    maximum_min_pred = min_pred_key(max(pred_real_max_min_close, key=min_pred_key))

    minimum_close_pred = close_pred_key(min(pred_real_max_min_close, key=close_pred_key))
    maximum_close_pred = close_pred_key(max(pred_real_max_min_close, key=close_pred_key))

    max_pred_step = 0.01
    max_pred_begin = max(math.floor(minimum_max_pred / max_pred_step) * max_pred_step, -0.05)
    max_pred_end = maximum_max_pred + max_pred_step

    min_pred_step = 0.01
    min_pred_begin = max(math.floor(minimum_min_pred / min_pred_step) * min_pred_step, -0.05)
    min_pred_end = maximum_min_pred + min_pred_step

    close_pred_step = 0.01
    close_pred_begin = max(math.floor(minimum_close_pred / close_pred_step)*close_pred_step, -0.05)
    close_pred_end = maximum_close_pred + close_pred_step

    for close_pred_thr in np.arange(close_pred_begin, close_pred_end, close_pred_step):
        for max_pred_thr in np.arange(max_pred_begin, max_pred_end, max_pred_step):
            for min_pred_thr in np.arange(min_pred_begin, min_pred_end, min_pred_step):
                filter_close = filter(lambda a: close_pred_key(a) >= close_pred_thr,
                                      pred_real_max_min_close)
                filter_max = filter(lambda a: max_pred_key(a) >= max_pred_thr, filter_close)
                filter_min = filter(lambda a: min_pred_key(a) >= min_pred_thr, filter_max)

                filtered = list(filter_min)

                if len(filtered) >= 10:
                    n_sig = len(filtered)

                    # TODO: implement and measure different TP and SL strategies

                    # tp strategies

                    close_pred_tp_count = sum(map(one_if_tp_at_close_pred_hit, filtered))
                    close_pred_tp_rate = close_pred_tp_count / n_sig
                    close_pred_total_profit = sum(map(tp_at_close_pred_profit, filtered))
                    close_pred_avg_profit = close_pred_total_profit / n_sig

                    max_pred_tp_count = sum(map(one_if_tp_at_max_pred_hit, filtered))
                    max_pred_tp_rate = max_pred_tp_count / n_sig
                    max_pred_total_profit = sum(map(tp_at_max_pred_profit, filtered))
                    max_pred_avg_profit = max_pred_total_profit / n_sig

                    one_if_close_pred_thr = get_one_if_tp_at_close_pred_thr_hit(close_pred_thr)
                    tp_at_close_pred_thr_profit = get_tp_at_close_pred_thr_profit(close_pred_thr)
                    close_pred_thr_tp_count = sum(map(one_if_close_pred_thr, filtered))
                    close_pred_thr_tp_rate = close_pred_thr_tp_count / n_sig
                    close_pred_thr_total_profit = sum(map(tp_at_close_pred_thr_profit, filtered))
                    close_pred_thr_avg_profit = close_pred_thr_total_profit / n_sig

                    one_if_max_pred_thr = get_one_if_tp_at_max_pred_thr_hit(max_pred_thr)
                    tp_at_max_pred_thr_profit = get_tp_at_max_pred_thr_profit(max_pred_thr)
                    max_pred_thr_tp_count = sum(map(one_if_max_pred_thr, filtered))
                    max_pred_thr_tp_rate = max_pred_thr_tp_count / n_sig
                    max_pred_thr_total_profit = sum(map(tp_at_max_pred_thr_profit, filtered))
                    max_pred_thr_avg_profit = max_pred_thr_total_profit / n_sig

                    # hit rates
                    max_pred_hit_count = sum(map(one_if_max_pred_hit, filtered))
                    max_pred_hit_rate = max_pred_hit_count / n_sig

                    min_pred_hit_count = sum(map(one_if_min_pred_hit, filtered))
                    min_pred_hit_rate = min_pred_hit_count / n_sig

                    close_pred_hit_count = sum(map(one_if_close_pred_hit, filtered))
                    close_pred_hit_rate = close_pred_hit_count / n_sig

                    # avg real max-min-close
                    real_max_sum = sum(map(max_real_key, filtered))
                    real_max_avg = real_max_sum / n_sig

                    real_min_sum = sum(map(min_real_key, filtered))
                    real_min_avg = real_min_sum / n_sig

                    real_close_sum = sum(map(close_real_key, filtered))
                    real_close_avg = real_close_sum / n_sig

                    print('cl_ma_mi {:<+5.2f} {:<+5.2f} {:<+5.2f} '
                          'n_sig {:<4} '
                          'cp_avgp {:<+8.5f} '
                          'mp_avgp {:<+8.5f} '
                          'cpt_avgp {:<+8.5f} '
                          'mpt_avgp {:<+8.5f} '
                          'cp_hr {:<+8.5f} '
                          'mp_hr {:<+8.5f} '
                          'max_avg {:<+8.5f} '
                          'close_avg {:<+8.5f} '
                          '                  '.format(close_pred_thr,
                                                      max_pred_thr,
                                                      min_pred_thr,
                                                      n_sig,
                                                      close_pred_avg_profit,
                                                      max_pred_avg_profit,
                                                      close_pred_thr_avg_profit,
                                                      max_pred_thr_avg_profit,
                                                      close_pred_hit_rate,
                                                      max_pred_hit_rate,
                                                      real_max_avg,
                                                      real_close_avg))

    # TODO: give some scores to model for some determined threshold levels

    print()
    for a in pred_real_max_min_close_profit[-10:]:
        print(a)


# VALIDATION PERFORMANCE

print()
print('++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++')
print()


try:
    pred_max = bst_max.predict(dval_max, ntree_limit=max_best_ntree_limit)
    pred_min = bst_min.predict(dval_min, ntree_limit=min_best_ntree_limit)
    pred_close = bst_close.predict(dval_close, ntree_limit=close_best_ntree_limit)
except XGBoostError:
    pred_max = bst_max.predict(dval_max)
    pred_min = bst_min.predict(dval_min)
    pred_close = bst_close.predict(dval_close)


pred_real_max = list(zip(pred_max, y_max_val))
pred_real_min = list(zip(pred_min, y_min_val))
pred_real_close = list(zip(pred_close, y_close_val))


analyze_output(pred_real_max, pred_real_min, pred_real_close)


# TEST PERFORMANCE

print()
print('++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++')
print()

try:
    pred_max = bst_max.predict(dval_max, ntree_limit=max_best_ntree_limit)
    pred_min = bst_min.predict(dval_min, ntree_limit=min_best_ntree_limit)
    pred_close = bst_close.predict(dval_close, ntree_limit=close_best_ntree_limit)
except XGBoostError:
    pred_max = bst_max.predict(dval_max)
    pred_min = bst_min.predict(dval_min)
    pred_close = bst_close.predict(dval_close)


pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))
pred_real_close = list(zip(pred_close, y_close_test))

analyze_output(pred_real_max, pred_real_min, pred_real_close)


# REAL TEST

print()
print('++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++')
print()

real_test_num_days = 3
real_test_start_date = dataset_end_date
real_test_end_date = real_test_start_date + seconds(days=real_test_num_days)


dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=real_test_start_date,
                                           end_date=real_test_end_date,
                                           feature_functions=feature_functions,
                                           value_function=max_price_value_func,
                                           cache_and_get_cached=True,
                                           feature_functions_key=feature_funcs_name,
                                           value_function_key=max_price_value_func_name)

max_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                           pair=dataset_symbol,
                                                           start_date=real_test_start_date,
                                                           end_date=real_test_end_date,
                                                           feature_functions=dummy_feature_functions,
                                                           value_function=max_price_value_func,
                                                           cache_and_get_cached=True,
                                                           feature_functions_key=dummy_feature_functions_name,
                                                           value_function_key=max_price_value_func_name)

min_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                           pair=dataset_symbol,
                                                           start_date=real_test_start_date,
                                                           end_date=real_test_end_date,
                                                           feature_functions=dummy_feature_functions,
                                                           value_function=min_price_value_func,
                                                           cache_and_get_cached=True,
                                                           feature_functions_key=dummy_feature_functions_name,
                                                           value_function_key=min_price_value_func_name)

close_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                             pair=dataset_symbol,
                                                             start_date=real_test_start_date,
                                                             end_date=real_test_end_date,
                                                             feature_functions=dummy_feature_functions,
                                                             value_function=close_price_value_func,
                                                             cache_and_get_cached=True,
                                                             feature_functions_key=dummy_feature_functions_name,
                                                             value_function_key=close_price_value_func_name)

X = dataset.get_numpy_feature_matrix()
y_max = max_price_value_dataset.get_numpy_value_array()
y_min = min_price_value_dataset.get_numpy_value_array()
y_close = close_price_value_dataset.get_numpy_value_array()

drealtest_max = xgb.DMatrix(X, label=y_max, feature_names=feature_names)

drealtest_min = xgb.DMatrix(X, label=y_min, feature_names=feature_names)

drealtest_close = xgb.DMatrix(X, label=y_close, feature_names=feature_names)

try:
    pred_max = bst_max.predict(drealtest_max, ntree_limit=max_best_ntree_limit)
    pred_min = bst_min.predict(drealtest_min, ntree_limit=min_best_ntree_limit)
    pred_close = bst_close.predict(drealtest_close, ntree_limit=close_best_ntree_limit)
except XGBoostError:
    pred_max = bst_max.predict(drealtest_max)
    pred_min = bst_min.predict(drealtest_min)
    pred_close = bst_close.predict(drealtest_close)

pred_real_max = list(zip(pred_max, y_max))
pred_real_min = list(zip(pred_min, y_min))
pred_real_close = list(zip(pred_close, y_close))

analyze_output(pred_real_max, pred_real_min, pred_real_close)
