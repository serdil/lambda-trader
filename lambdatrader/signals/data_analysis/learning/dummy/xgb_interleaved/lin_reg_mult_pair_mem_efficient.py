from collections import namedtuple

from datetime import datetime
from operator import itemgetter

import numpy as np
import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX, ExchangeEnum
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseReturn, MaxReturn, MinReturn
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory as fsf
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output
from lambdatrader.utilities.utils import seconds

all_symbols = set(SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs())

# symbols = ['BTC_ETH']
# symbols = ['BTC_VIA', 'BTC_SC', 'BTC_ETH']
# symbols = ['BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_ETH']
# symbols = ['BTC_LTC', 'BTC_ETC', 'BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_ETH']
symbols = sorted(list(all_symbols))

num_candles = 48

day_offset = 12
days = 200

feature_set = fsf.get_small()

val_ratio = 0.8
test_ratio = 0.9

CMMValueSet = namedtuple('CMMValueSet', ['value_set', 'close_name', 'max_name', 'min_name'])


def get_cmm_value_set():
    close_return= CloseReturn(n_candles=num_candles, period=M5)
    max_return = MaxReturn(n_candles=num_candles, period=M5)
    min_return = MinReturn(n_candles=num_candles, period=M5)
    vs = DFFeatureSet(features=[close_return, max_return, min_return])
    return CMMValueSet(value_set=vs,
                       close_name=close_return.name,
                       max_name=max_return.name,
                       min_name=min_return.name)


def get_train_val_test_Xs_ys_feature_names():
    market_info = BacktestingMarketInfo(
        candlestick_store=ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))
    latest_market_date = market_info.get_max_pair_end_time()

    dataset_start_date = latest_market_date - seconds(days=day_offset + days)

    dataset_end_date = latest_market_date - seconds(days=day_offset)

    print('start_date: {} end_date: {}'.format(datetime.utcfromtimestamp(dataset_start_date),
                                               datetime.utcfromtimestamp(dataset_end_date)))
    print()
    cmm_vs = get_cmm_value_set()
    X, y_close, y_max, f_names= (DFDataset
                                 .compute_interleaved(pairs=symbols,
                                                      feature_set=feature_set,
                                                      value_set=cmm_vs.value_set,
                                                      start_date=dataset_start_date,
                                                      end_date=dataset_end_date,
                                                      error_on_missing=False)
                                 .add_feature_values()
                                 .add_value_values(value_name=cmm_vs.close_name)
                                 .add_value_values(value_name=cmm_vs.max_name)
                                 .add_feature_names()
                                 .get())

    n_samples = len(X)
    gap = num_candles * len(symbols)

    validation_split_ind = int(n_samples * val_ratio)
    test_split_ind = int(n_samples * test_ratio)

    X_train = X[:validation_split_ind - gap]
    y_close_train = y_close[:validation_split_ind - gap]
    y_max_train = y_max[:validation_split_ind - gap]

    X_val = X[validation_split_ind:test_split_ind - gap]
    y_close_val = y_close[validation_split_ind:test_split_ind - gap]
    y_max_val = y_max[validation_split_ind:test_split_ind - gap]

    X_test = X[test_split_ind:]
    y_close_test = y_close[test_split_ind:]
    y_max_test = y_max[test_split_ind:]

    return X_train, y_close_train, y_max_train, \
           X_val, y_close_val, y_max_val, \
           X_test, y_close_test, y_max_test, f_names


def get_test_X_ys():
    X_train, y_close_train, y_max_train,\
    X_val, y_close_val, y_max_val, \
    X_test, y_close_test, y_max_test, f_names = get_train_val_test_Xs_ys_feature_names()
    return X_test, y_close_test, y_max_test


def get_close_dmatrix():
    X_train, y_close_train, y_max_train, X_val, y_close_val, y_max_val, X_test, y_close_test, \
    y_max_test, feature_names = get_train_val_test_Xs_ys_feature_names()

    dtrain_close = xgb.DMatrix(X_train, label=y_close_train, feature_names=feature_names)
    dval_close = xgb.DMatrix(X_val, label=y_close_val, feature_names=feature_names)
    dtest_close = xgb.DMatrix(X_test, label=y_close_test, feature_names=feature_names)

    return dtrain_close, dval_close, dtest_close


def get_max_dmatrix():
    X_train, y_close_train, y_max_train, X_val, y_close_val, y_max_val, X_test, y_close_test, \
    y_max_test, feature_names = get_train_val_test_Xs_ys_feature_names()

    dtrain_max = xgb.DMatrix(X_train, label=y_max_train, feature_names=feature_names)
    dval_max = xgb.DMatrix(X_val, label=y_max_val, feature_names=feature_names)
    dtest_max = xgb.DMatrix(X_test, label=y_max_test, feature_names=feature_names)

    return dtrain_max, dval_max, dtest_max


params = {
    'silent': 1,
    'booster': 'gblinear',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.01,
    'gamma': 0,
    'max_depth': 2,
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
    'updater': 'shotgun'
}

num_round = 1000
early_stopping_rounds = 10


def train_close():
    dtrain_close, dval_close, dtest_close = get_close_dmatrix()
    watchlist_close = [(dtrain_close, 'train_close'), (dtest_close, 'test_close'),
                       (dval_close, 'val_close')]

    bst_close = xgb.train(params=params, dtrain=dtrain_close, num_boost_round=num_round,
                          evals=watchlist_close, early_stopping_rounds=early_stopping_rounds)

    close_best_ntree_limit = bst_close.best_ntree_limit
    feature_importances_close = bst_close.get_fscore()

    print()
    print('feature importances close:')
    for f_name, imp in list(reversed(sorted(feature_importances_close.items(), key=itemgetter(1))))[
                       :10]:
        print(f_name, ':', imp)

    try:
        pred_close = bst_close.predict(dtest_close, ntree_limit=close_best_ntree_limit)
    except XGBoostError:
        pred_close = bst_close.predict(dtest_close)
    return pred_close


def train_max():
    max_params = params.copy()
    override_params = {
        'eta': 0.1
    }
    max_params.update(override_params)

    dtrain_max, dval_max, dtest_max = get_max_dmatrix()
    watchlist_max = [(dtrain_max, 'train_max'), (dtest_max, 'test_max'),
                       (dval_max, 'val_max')]

    bst_max = xgb.train(params=max_params, dtrain=dtrain_max, num_boost_round=num_round,
                          evals=watchlist_max, early_stopping_rounds=early_stopping_rounds)

    max_best_ntree_limit = bst_max.best_ntree_limit
    feature_importances_max = bst_max.get_fscore()

    print()
    print('feature importances max:')
    for f_name, imp in list(reversed(sorted(feature_importances_max.items(), key=itemgetter(1))))[
                       :10]:
        print(f_name, ':', imp)

    try:
        pred_max = bst_max.predict(dtest_max, ntree_limit=max_best_ntree_limit)
    except XGBoostError:
        pred_max = bst_max.predict(dtest_max)
    return pred_max


pred_close = train_close()
pred_max = train_max()

print()
print('++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++++++TEST++++')
print()

pred_min = np.zeros(len(pred_close))
y_min_test = np.zeros(len(pred_close))

X_test, y_close_test, y_max_test = get_test_X_ys()

pred_real_close = list(zip(pred_close, y_close_test))
pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))

analyze_output(pred_real_close=pred_real_close,
               pred_real_max=pred_real_max,
               pred_real_min=pred_real_min)
