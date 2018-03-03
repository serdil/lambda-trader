# get max prices, min prices and close prices for 4 hours
# train model to predict max price in 4 hours
# train model to predict min price in 4 hours
# train model to predict close price in 4 hours
# determine trading gains for different
# (max_price_threshold, min_price_threshold, close_price_threshold) pairs.

from operator import itemgetter

import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.signals.data_analysis.factories import FeatureSets
from lambdatrader.signals.data_analysis.learning.dummy.dummy_utils_dummy import (
    get_dataset_info,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output

symbol = 'BTC_ETH'

num_candles = 48

day_offset = 120
days = 500

feature_set = FeatureSets.get_small()

ds_info = get_dataset_info(symbol=symbol, day_offset=day_offset, days=days, feature_set=feature_set)

X, y_close, y_max, y_min, feature_names = ds_info.x, ds_info.y_close, \
                                          ds_info.y_max, ds_info.y_min, ds_info.feature_names

print('created/loaded dataset\n')

train_ratio = 0.7
validation_ratio = 0.85
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
    pred_max = bst_max.predict(dtest_max, ntree_limit=max_best_ntree_limit)
    pred_min = bst_min.predict(dtest_min, ntree_limit=min_best_ntree_limit)
    pred_close = bst_close.predict(dtest_close, ntree_limit=close_best_ntree_limit)
except XGBoostError:
    pred_max = bst_max.predict(dtest_max)
    pred_min = bst_min.predict(dtest_min)
    pred_close = bst_close.predict(dtest_close)


pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))
pred_real_close = list(zip(pred_close, y_close_test))

analyze_output(pred_real_max, pred_real_min, pred_real_close)


# REAL TEST

# print()
# print('++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++++++REAL_TEST++++')
# print()
#
# real_test_num_days = 7
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
# try:
#     pred_max = bst_max.predict(drealtest_max, ntree_limit=max_best_ntree_limit)
#     pred_min = bst_min.predict(drealtest_min, ntree_limit=min_best_ntree_limit)
#     pred_close = bst_close.predict(drealtest_close, ntree_limit=close_best_ntree_limit)
# except XGBoostError:
#     pred_max = bst_max.predict(drealtest_max)
#     pred_min = bst_min.predict(drealtest_min)
#     pred_close = bst_close.predict(drealtest_close)
#
# pred_real_max = list(zip(pred_max, y_max))
# pred_real_min = list(zip(pred_min, y_min))
# pred_real_close = list(zip(pred_close, y_close))
#
# analyze_output(pred_real_max, pred_real_min, pred_real_close)
