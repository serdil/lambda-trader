from collections import namedtuple

from datetime import datetime
from operator import itemgetter

import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseReturn, MaxReturn, MinReturn
from lambdatrader.utilities.utils import seconds

CMMValueSet = namedtuple('CMMValueSet', ['value_set', 'close_name', 'max_name', 'min_name'])


def get_cmm_value_set(num_candles):
    close_return = CloseReturn(n_candles=num_candles, period=M5)
    max_return = MaxReturn(n_candles=num_candles, period=M5)
    min_return = MinReturn(n_candles=num_candles, period=M5)
    vs = DFFeatureSet(features=[close_return, max_return, min_return])
    return CMMValueSet(value_set=vs, close_name=close_return.name, max_name=max_return.name,
                       min_name=min_return.name)


def get_train_val_test_Xs_ys_feature_names(symbols, feature_set, val_ratio, test_ratio, num_candles,
                                           day_offset, days):
    market_info = BacktestingMarketInfo(
        candlestick_store=ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))
    latest_market_date = market_info.get_max_pair_end_time()

    dataset_start_date = latest_market_date - seconds(days=day_offset + days)

    dataset_end_date = latest_market_date - seconds(days=day_offset)

    print('start_date: {} end_date: {}'.format(datetime.utcfromtimestamp(dataset_start_date),
                                               datetime.utcfromtimestamp(dataset_end_date)))
    print()
    cmm_vs = get_cmm_value_set(num_candles=num_candles)
    X, y_close, y_max, f_names = (
    DFDataset.compute_interleaved(pairs=symbols, feature_set=feature_set,
                                  value_set=cmm_vs.value_set, start_date=dataset_start_date,
                                  end_date=dataset_end_date,
                                  error_on_missing=False).add_feature_values().add_value_values(
        value_name=cmm_vs.close_name).add_value_values(
        value_name=cmm_vs.max_name).add_feature_names().get())

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

    return X_train, y_close_train, y_max_train, X_val, y_close_val, y_max_val, X_test, \
           y_close_test, y_max_test, f_names


def get_test_X_ys(symbols, feature_set, val_ratio, test_ratio, num_candles, day_offset, days,
                  **kwargs):
    X_train, y_close_train, y_max_train, X_val, y_close_val, y_max_val, X_test, y_close_test, \
    y_max_test, f_names = get_train_val_test_Xs_ys_feature_names(
        symbols=symbols, val_ratio=val_ratio, test_ratio=test_ratio, num_candles=num_candles,
        day_offset=day_offset, days=days, feature_set=feature_set)
    return X_test, y_close_test, y_max_test


def get_close_dmatrix(symbols, feature_set, val_ratio, test_ratio, num_candles, day_offset, days):
    X_train, y_close_train, y_max_train, X_val, y_close_val, y_max_val, X_test, y_close_test, \
    y_max_test, feature_names = get_train_val_test_Xs_ys_feature_names(
        symbols=symbols, val_ratio=val_ratio, test_ratio=test_ratio, num_candles=num_candles,
        day_offset=day_offset, days=days, feature_set=feature_set)

    dtrain_close = xgb.DMatrix(X_train, label=y_close_train, feature_names=feature_names)
    dval_close = xgb.DMatrix(X_val, label=y_close_val, feature_names=feature_names)
    dtest_close = xgb.DMatrix(X_test, label=y_close_test, feature_names=feature_names)

    return dtrain_close, dval_close, dtest_close


def get_max_dmatrix(symbols, feature_set, val_ratio, test_ratio, num_candles, day_offset, days):
    X_train, y_close_train, y_max_train, X_val, y_close_val, y_max_val, X_test, y_close_test, \
    y_max_test, feature_names = get_train_val_test_Xs_ys_feature_names(
        symbols=symbols, val_ratio=val_ratio, test_ratio=test_ratio, num_candles=num_candles,
        day_offset=day_offset, days=days, feature_set=feature_set)

    dtrain_max = xgb.DMatrix(X_train, label=y_max_train, feature_names=feature_names)
    dval_max = xgb.DMatrix(X_val, label=y_max_val, feature_names=feature_names)
    dtest_max = xgb.DMatrix(X_test, label=y_max_test, feature_names=feature_names)

    return dtrain_max, dval_max, dtest_max


def train_close(params, num_rounds, early_stopping_rounds, symbols, feature_set, val_ratio,
                test_ratio, num_candles, day_offset, days):
    dtrain_close, dval_close, dtest_close = get_close_dmatrix(symbols=symbols,
                                                              feature_set=feature_set,
                                                              val_ratio=val_ratio,
                                                              test_ratio=test_ratio,
                                                              num_candles=num_candles,
                                                              day_offset=day_offset, days=days)
    return _train_close_with_dmatrices(params, num_rounds, early_stopping_rounds,
                                       (dtrain_close, dval_close, dtest_close,))


def train_close_from_saved(params, num_rounds, early_stopping_rounds, symbols, feature_set,
                           val_ratio, test_ratio, num_candles, day_offset, days):
    valr = int(val_ratio * 10)
    testr = int(test_ratio * 10)
    from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.dmatrix_save_load_util\
        import \
        load_close_dmatrix
    dmatrices = load_close_dmatrix(num_candles=num_candles, candle_period=M5,
                                   feature_set=feature_set, num_days=days, days_offset=day_offset,
                                   symbols=symbols, valr=valr, testr=testr)
    return _train_close_with_dmatrices(params, num_rounds, early_stopping_rounds, dmatrices)


def _train_close_with_dmatrices(params, num_rounds, early_stopping_rounds, dmatrices):
    dtrain_close, dval_close, dtest_close = dmatrices
    watchlist_close = [(dtrain_close, 'train_close'), (dtest_close, 'test_close'),
                       (dval_close, 'val_close')]

    bst_close = xgb.train(params=params, dtrain=dtrain_close, num_boost_round=num_rounds,
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


def train_max(params, num_rounds, early_stopping_rounds, symbols, feature_set, val_ratio,
              test_ratio, num_candles, day_offset, days):
    dtrain_max, dval_max, dtest_max = get_max_dmatrix(symbols=symbols, feature_set=feature_set,
                                                      val_ratio=val_ratio, test_ratio=test_ratio,
                                                      num_candles=num_candles,
                                                      day_offset=day_offset, days=days)
    return _train_max_with_dmatrices(params, num_rounds, early_stopping_rounds,
                                     (dtrain_max, dval_max, dtest_max,))


def train_max_from_saved(params, num_rounds, early_stopping_rounds, symbols, feature_set, val_ratio,
                         test_ratio, num_candles, day_offset, days):
    valr = int(val_ratio * 10)
    testr = int(test_ratio * 10)
    from lambdatrader.signals.data_analysis.learning.dummy.xgb_interleaved.dmatrix_save_load_util\
        import \
        load_max_dmatrix
    dmatrices = load_max_dmatrix(num_candles=num_candles, candle_period=M5, feature_set=feature_set,
                                 num_days=days, days_offset=day_offset, symbols=symbols, valr=valr,
                                 testr=testr)
    return _train_max_with_dmatrices(params, num_rounds, early_stopping_rounds, dmatrices)


def _train_max_with_dmatrices(params, num_rounds, early_stopping_rounds, dmatrices):
    dtrain_max, dval_max, dtest_max = dmatrices
    watchlist_max = [(dtrain_max, 'train_max'), (dtest_max, 'test_max'), (dval_max, 'val_max')]

    bst_max = xgb.train(params=params, dtrain=dtrain_max, num_boost_round=num_rounds,
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
