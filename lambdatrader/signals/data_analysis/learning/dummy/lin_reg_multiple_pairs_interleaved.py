
from operator import itemgetter

import xgboost as xgb
from pandas.core.base import DataError
from xgboost.core import XGBoostError

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory
from lambdatrader.signals.data_analysis.learning.dummy.dummy_utils_dummy import (
    get_dataset_info,
)
from lambdatrader.signals.data_analysis.learning.dummy.np_interleave_util import (
    interleave_2d, interleave_1d,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output

all_symbols = set(SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs())

# symbols = ['BTC_ETH']
# symbols = ['BTC_VIA', 'BTC_SC', 'BTC_ETH']
# symbols = ['BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_ETH']
# symbols = ['BTC_LTC', 'BTC_ETC', 'BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_ETH']
symbols = sorted(list(all_symbols))

num_candles = 48

day_offset = 120
days = 120

feature_set = DFFeatureSetFactory.get_small()

ds_infos = []


for symbol in symbols:
    print('loading {}'.format(symbol))
    try:
        ds_info = get_dataset_info(symbol=symbol, day_offset=day_offset, days=days, feature_set=feature_set)
        ds_infos.append(ds_info)
    except DataError:
        print('DataError: {}'.format(symbol))

xs = [ds_info.x for ds_info in ds_infos]
y_closes = [ds_info.y_close for ds_info in ds_infos]
y_maxs = [ds_info.y_max for ds_info in ds_infos]
y_mins = [ds_info.y_min for ds_info in ds_infos]

X = interleave_2d(xs, num_rows=num_candles)
y_close = interleave_1d(y_closes, num_rows=num_candles)
y_max = interleave_1d(y_maxs, num_rows=num_candles)
y_min = interleave_1d(y_mins, num_rows=num_candles)

feature_names = ds_infos[0].feature_names

n_samples = len(X)
gap = num_candles * len(symbols)

val_ratio = 0.8
test_ratio = 0.9

validation_split_ind = int(n_samples * val_ratio)
test_split_ind = int(n_samples * test_ratio)


X_train = X[:validation_split_ind - gap]
y_max_train = y_max[:validation_split_ind - gap]
y_min_train = y_min[:validation_split_ind - gap]
y_close_train = y_close[:validation_split_ind - gap]


X_val = X[validation_split_ind:test_split_ind-gap]
y_max_val = y_max[validation_split_ind:test_split_ind-gap]
y_min_val = y_min[validation_split_ind:test_split_ind-gap]
y_close_val = y_close[validation_split_ind:test_split_ind-gap]


X_test = X[test_split_ind:]
y_max_test = y_max[test_split_ind:]
y_min_test = y_min[test_split_ind:]
y_close_test = y_close[test_split_ind:]

print('test set size:', len(X_test))
print('test set size / 288:', len(X_test) / 288)

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

# print()
# print('++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++++++VALIDATION++++')
# print()
#
#
# try:
#     pred_max = bst_max.predict(dval_max, ntree_limit=max_best_ntree_limit)
#     pred_min = bst_min.predict(dval_min, ntree_limit=min_best_ntree_limit)
#     pred_close = bst_close.predict(dval_close, ntree_limit=close_best_ntree_limit)
# except XGBoostError:
#     pred_max = bst_max.predict(dval_max)
#     pred_min = bst_min.predict(dval_min)
#     pred_close = bst_close.predict(dval_close)
#
#
# pred_real_max = list(zip(pred_max, y_max_val))
# pred_real_min = list(zip(pred_min, y_min_val))
# pred_real_close = list(zip(pred_close, y_close_val))
#
#
# analyze_output(pred_real_max, pred_real_min, pred_real_close)


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
