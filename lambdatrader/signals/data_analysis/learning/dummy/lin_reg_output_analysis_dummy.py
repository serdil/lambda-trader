
# get max prices, min prices and close prices for 1 hour
# train model to predict max price in 1 hour
# train model to predict min price in 1 hour
# determine trading gains for different (max_price_threshold, min_price_threshold) pairs.
import numpy as np
import xgboost as xgb

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
dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*365)
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


dataset_symbol = 'BTC_ETC'

dummy_feature_functions = list(get_dummy_feature_func_set())
dummy_feature_functions_name = 'dummy'

feature_functions = list(get_small_feature_func_set_with_indicators())
feature_funcs_name = 'with_ind'

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
    'booster': 'gbtree',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.1,
    'gamma': 0,
    'max_depth': 2,
    'min_child_weight': 1,
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

watchlist_max = [(dtrain_max, 'train_max'), (dval_max, 'val_max'), (dtest_max, 'test_max')]
watchlist_min = [(dtrain_min, 'train_min'), (dval_min, 'val_min'), (dtest_min, 'test_min')]
watchlist_close = [(dtrain_close, 'train_close'),
                   (dval_close, 'val_close'), (dtest_close, 'test_close')]


num_round = 10000
early_stopping_rounds = 200

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


max_pred_key = lambda a: a[0][0]
min_pred_key = lambda a: a[1][0]
close_pred_key = lambda a: a[2][0]

max_real_key = lambda a: a[0][1]
min_real_key = lambda a: a[1][1]
close_real_key = lambda a: a[2][1]

tp_level = 1.0
compute_profit = lambda a: close_pred_key(a) * tp_level \
    if close_pred_key(a) > 0 and max_real_key(a) >= close_pred_key(a) * tp_level \
    else close_real_key(a)
one_for_tp_hit_else_zero = lambda a: 1 \
    if close_pred_key(a) > 0 and max_real_key(a) >= close_pred_key(a) * tp_level \
    else 0


max_best_ntree_limit = bst_max.best_ntree_limit
min_best_ntree_limit = bst_min.best_ntree_limit
close_best_ntree_limit = bst_close.best_ntree_limit


# VALIDATION PERFORMANCE

print()
print('++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++')
print()

pred_max = bst_max.predict(dval_max, ntree_limit=max_best_ntree_limit)
pred_min = bst_min.predict(dval_min, ntree_limit=min_best_ntree_limit)
pred_close = bst_close.predict(dval_close, ntree_limit=close_best_ntree_limit)

pred_real_max = list(zip(pred_max, y_max_val))
pred_real_min = list(zip(pred_min, y_min_val))
pred_real_close = list(zip(pred_close, y_close_val))

pred_real_max_min_close = list(zip(pred_real_max, pred_real_min, pred_real_close))

profits = [compute_profit(a) for a in pred_real_max_min_close]

pred_real_max_min_close_profit = list(zip(pred_real_max_min_close, profits))

minimum_max_pred = max_pred_key(min(pred_real_max_min_close, key=max_pred_key))
maximum_max_pred = max_pred_key(max(pred_real_max_min_close, key=max_pred_key))

minimum_min_pred = min_pred_key(min(pred_real_max_min_close, key=min_pred_key))
maximum_min_pred = min_pred_key(max(pred_real_max_min_close, key=min_pred_key))

minimum_close_pred = close_pred_key(min(pred_real_max_min_close, key=close_pred_key))
maximum_close_pred = close_pred_key(max(pred_real_max_min_close, key=close_pred_key))

max_pred_step = 0.01
max_pred_begin = 0.00
max_pred_end = maximum_max_pred+max_pred_step

min_pred_step = 0.01
min_pred_begin = -0.05
min_pred_end = maximum_min_pred+min_pred_step

close_pred_step = 0.01
close_pred_begin = 0.00
close_pred_end = maximum_close_pred+close_pred_step

for close_pred_threshold in np.arange(close_pred_begin, close_pred_end, close_pred_step):
    for max_pred_threshold in np.arange(max_pred_begin, max_pred_end, max_pred_step):
        for min_pred_threshold in np.arange(min_pred_begin, min_pred_end, min_pred_step):
            filter_close = filter(lambda a: close_pred_key(a) >= close_pred_threshold, pred_real_max_min_close)
            filter_max = filter(lambda a: max_pred_key(a) >= max_pred_threshold, filter_close)
            filter_min = filter(lambda a: min_pred_key(a) >= min_pred_threshold, filter_max)

            filtered = list(filter_min)

            if filtered:
                n_sig = len(filtered)
                total_profit = sum([compute_profit(a) for a in filtered])

                avg_profit = total_profit / n_sig

                true_pos = sum([one_for_tp_hit_else_zero(a) for a in filtered])

                min_sum_score = max_pred_threshold + min_pred_threshold + close_pred_threshold
                avg_sum_score = sum([max_pred_key(a) + min_pred_key(a) + close_pred_key(a)
                                     for a in filtered]) / n_sig

                print('max_min_close_thr {:<+8.5f} {:<+8.5f} {:<+8.5f} n_sig {:<4} true_pos {:<4} total_profit {:<+7.4f} avg_profit {:<+8.5f} min_sum_s {:<+8.5f}, avg_s_s {:<+8.5f}'
                      .format(max_pred_threshold, min_pred_threshold, close_pred_threshold, n_sig, true_pos, total_profit, avg_profit, min_sum_score, avg_sum_score))


for a in pred_real_max_min_close_profit[-10:]:
    print(a)


# TEST PERFORMANCE

print()
print('++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++')
print()

pred_max = bst_max.predict(dtest_max, ntree_limit=max_best_ntree_limit)
pred_min = bst_min.predict(dtest_min, ntree_limit=min_best_ntree_limit)
pred_close = bst_close.predict(dtest_close, ntree_limit=close_best_ntree_limit)

pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))
pred_real_close = list(zip(pred_close, y_close_test))

pred_real_max_min_close = list(zip(pred_real_max, pred_real_min, pred_real_close))

profits = [compute_profit(a) for a in pred_real_max_min_close]

pred_real_max_min_close_profit = list(zip(pred_real_max_min_close, profits))

maximum_max_pred = max_pred_key(max(pred_real_max_min_close, key=max_pred_key))

maximum_min_pred = min_pred_key(max(pred_real_max_min_close, key=min_pred_key))

maximum_close_pred = close_pred_key(max(pred_real_max_min_close, key=close_pred_key))

max_pred_step = 0.01
max_pred_begin = 0.00
max_pred_end = maximum_max_pred+max_pred_step

min_pred_step = 0.01
min_pred_begin = -0.05
min_pred_end = maximum_min_pred+min_pred_step

close_pred_step = 0.01
close_pred_begin = 0.00
close_pred_end = maximum_close_pred+close_pred_step

for close_pred_threshold in np.arange(close_pred_begin, close_pred_end, close_pred_step):
    for max_pred_threshold in np.arange(max_pred_begin, max_pred_end, max_pred_step):
        for min_pred_threshold in np.arange(min_pred_begin, min_pred_end, min_pred_step):
            filter_close = filter(lambda a: close_pred_key(a) >= close_pred_threshold, pred_real_max_min_close)
            filter_max = filter(lambda a: max_pred_key(a) >= max_pred_threshold, filter_close)
            filter_min = filter(lambda a: min_pred_key(a) >= min_pred_threshold, filter_max)

            filtered = list(filter_min)

            if filtered:
                n_sig = len(filtered)
                total_profit = sum([compute_profit(a) for a in filtered])

                avg_profit = total_profit / n_sig

                true_pos = sum([one_for_tp_hit_else_zero(a) for a in filtered])

                min_sum_score = max_pred_threshold + min_pred_threshold + close_pred_threshold
                avg_sum_score = sum([max_pred_key(a) + min_pred_key(a) + close_pred_key(a)
                                     for a in filtered]) / n_sig

                print('max_min_close_thr {:<+8.5f} {:<+8.5f} {:<+8.5f} n_sig {:<4} true_pos {:<4} total_profit {:<+7.4f} avg_profit {:<+8.5f} min_sum_s {:<+8.5f}, avg_s_s {:<+8.5f}'
                      .format(max_pred_threshold, min_pred_threshold, close_pred_threshold, n_sig, true_pos, total_profit, avg_profit, min_sum_score, avg_sum_score))


for a in pred_real_max_min_close_profit[-10:]:
    print(a)


#
# # REAL TEST
#
# real_test_num_days = 30
# real_test_start_date = dataset_end_date
# real_test_end_date = real_test_start_date + seconds(days=real_test_num_days)
#
#
# dataset = create_pair_dataset_from_history(market_info=market_info,
#                                            pair=dataset_symbol,
#                                            start_date=real_test_start_date,
#                                            end_date=real_test_end_date,
#                                            feature_functions=feature_functions,
#                                            value_function=max_price_value_func,
#                                            cache_and_get_cached=True,
#                                            feature_functions_key=feature_funcs_name,
#                                            value_function_key=max_price_value_func_name)
#
# max_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
#                                                            pair=dataset_symbol,
#                                                            start_date=real_test_start_date,
#                                                            end_date=real_test_end_date,
#                                                            feature_functions=dummy_feature_functions,
#                                                            value_function=max_price_value_func,
#                                                            cache_and_get_cached=True,
#                                                            feature_functions_key=dummy_feature_functions_name,
#                                                            value_function_key=max_price_value_func_name)
#
# min_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
#                                                            pair=dataset_symbol,
#                                                            start_date=real_test_start_date,
#                                                            end_date=real_test_end_date,
#                                                            feature_functions=dummy_feature_functions,
#                                                            value_function=min_price_value_func,
#                                                            cache_and_get_cached=True,
#                                                            feature_functions_key=dummy_feature_functions_name,
#                                                            value_function_key=min_price_value_func_name)
#
# close_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
#                                                              pair=dataset_symbol,
#                                                              start_date=real_test_start_date,
#                                                              end_date=real_test_end_date,
#                                                              feature_functions=dummy_feature_functions,
#                                                              value_function=close_price_value_func,
#                                                              cache_and_get_cached=True,
#                                                              feature_functions_key=dummy_feature_functions_name,
#                                                              value_function_key=close_price_value_func_name)
#
# X = dataset.get_numpy_feature_matrix()
# y_max = max_price_value_dataset.get_numpy_value_array()
# y_min = min_price_value_dataset.get_numpy_value_array()
# y_close = close_price_value_dataset.get_numpy_value_array()
#
# drealtest_max = xgb.DMatrix(X, label=y_max, feature_names=feature_names)
#
# drealtest_min = xgb.DMatrix(X, label=y_min, feature_names=feature_names)
#
# drealtest_close = xgb.DMatrix(X, label=y_close, feature_names=feature_names)
#
# pred_max = bst_max.predict(drealtest_max, ntree_limit=max_best_ntree_limit)
# pred_min = bst_min.predict(drealtest_min, ntree_limit=min_best_ntree_limit)
# pred_close = bst_close.predict(drealtest_close, ntree_limit=close_best_ntree_limit)
#
# pred_real_max = list(zip(pred_max, y_max_val))
# pred_real_min = list(zip(pred_min, y_min_val))
# pred_real_close = list(zip(pred_close, y_close_val))
#
# pred_real_max_min_close = list(zip(pred_real_max, pred_real_min, pred_real_close))
#
# profits = [compute_profit(a) for a in pred_real_max_min_close]
#
# pred_real_max_min_close_profit = list(zip(pred_real_max_min_close, profits))
#
# maximum_max_pred = max_pred_key(max(pred_real_max_min_close, key=max_pred_key))
#
# maximum_min_pred = min_pred_key(max(pred_real_max_min_close, key=min_pred_key))
#
# maximum_close_pred = close_pred_key(max(pred_real_max_min_close, key=close_pred_key))
#
# max_pred_step = 0.01
# max_pred_begin = 0.00
# max_pred_end = maximum_max_pred+max_pred_step
#
# min_pred_step = 0.01
# min_pred_begin = -0.05
# min_pred_end = maximum_min_pred+min_pred_step
#
# close_pred_step = 0.01
# close_pred_begin = 0.00
# close_pred_end = maximum_close_pred+close_pred_step
#
# print()
# print('++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++')
# print('++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++')
# print('++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++')
# print()
#
#
# for close_pred_threshold in np.arange(close_pred_begin, close_pred_end, close_pred_step):
#     for max_pred_threshold in np.arange(max_pred_begin, max_pred_end, max_pred_step):
#         for min_pred_threshold in np.arange(min_pred_begin, min_pred_end, min_pred_step):
#             filter_close = filter(lambda a: close_pred_key(a) >= close_pred_threshold, pred_real_max_min_close)
#             filter_max = filter(lambda a: max_pred_key(a) >= max_pred_threshold, filter_close)
#             filter_min = filter(lambda a: min_pred_key(a) >= min_pred_threshold, filter_max)
#
#             filtered = list(filter_min)
#
#             if filtered:
#                 n_sig = len(filtered)
#                 total_profit = sum([compute_profit(a) for a in filtered])
#
#                 avg_profit = total_profit / n_sig
#
#                 true_pos = sum([one_for_tp_hit_else_zero(a) for a in filtered])
#
#                 min_sum_score = max_pred_threshold + min_pred_threshold + close_pred_threshold
#                 avg_sum_score = sum([max_pred_key(a) + min_pred_key(a) + close_pred_key(a)
#                                      for a in filtered]) / n_sig
#
#                 print('max_min_close_thr {:<+8.5f} {:<+8.5f} {:<+8.5f} n_sig {:<4} true_pos {:<4} total_profit {:<+7.4f} avg_profit {:<+8.5f} min_sum_s {:<+8.5f}, avg_s_s {:<+8.5f}'
#                       .format(max_pred_threshold, min_pred_threshold, close_pred_threshold, n_sig, true_pos, total_profit, avg_profit, min_sum_score, avg_sum_score))
#
#
# for a in pred_real_max_min_close_profit[-10:]:
#     print(a)
