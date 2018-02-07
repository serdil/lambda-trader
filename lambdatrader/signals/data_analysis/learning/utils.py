import numpy as np
import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.constants import M5
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set, get_dummy_feature_func_set,
)
from lambdatrader.signals.data_analysis.values import (
    make_cont_max_price_in_future, make_cont_min_price_in_future, make_cont_close_price_in_future,
)


def train_max_min_close_pred_lin_reg_model(market_info, 
                                           pair, 
                                           start_date, 
                                           end_date, 
                                           num_candles=48, 
                                           train_ratio=0.8, 
                                           candle_period=M5):
    feature_funcs = list(get_small_feature_func_set())
    dummy_feature_funcs = list(get_dummy_feature_func_set())
    max_price_value_func =  make_cont_max_price_in_future(num_candles=num_candles,
                                                          candle_period=candle_period)
    min_price_value_func = make_cont_min_price_in_future(num_candles=num_candles,
                                                         candle_period=candle_period)
    close_price_value_func = make_cont_close_price_in_future(num_candles=num_candles,
                                                             candle_period=candle_period)
    
    dataset = create_pair_dataset_from_history(market_info=market_info,
                                               pair=pair,
                                               start_date=start_date,
                                               end_date=end_date,
                                               feature_functions=feature_funcs,
                                               value_function=max_price_value_func,
                                               cache_and_get_cached=False)

    max_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                               pair=pair,
                                                               start_date=start_date,
                                                               end_date=end_date,
                                                               feature_functions=
                                                               dummy_feature_funcs,
                                                               value_function=max_price_value_func,
                                                               cache_and_get_cached=False)

    min_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                               pair=pair,
                                                               start_date=start_date,
                                                               end_date=end_date,
                                                               feature_functions=
                                                               dummy_feature_funcs,
                                                               value_function=min_price_value_func,
                                                               cache_and_get_cached=False)

    close_price_value_dataset = create_pair_dataset_from_history(market_info=market_info,
                                                                 pair=pair,
                                                                 start_date=start_date,
                                                                 end_date=end_date,
                                                                 feature_functions=
                                                                 dummy_feature_funcs,
                                                                 value_function=
                                                                 close_price_value_func,
                                                                 cache_and_get_cached=False)

    feature_names = dataset.get_first_feature_names()
    X = dataset.get_numpy_feature_matrix()
    y_max = max_price_value_dataset.get_numpy_value_array()
    y_min = min_price_value_dataset.get_numpy_value_array()
    y_close = close_price_value_dataset.get_numpy_value_array()

    gap = num_candles

    n_samples = len(X)

    validation_split_ind = int(train_ratio * n_samples)

    X_train = X[:validation_split_ind - gap]
    y_max_train = y_max[:validation_split_ind - gap]
    y_min_train = y_min[:validation_split_ind - gap]
    y_close_train = y_close[:validation_split_ind - gap]

    X_val = X[validation_split_ind:]
    y_max_val = y_max[validation_split_ind:]
    y_min_val = y_min[validation_split_ind:]
    y_close_val = y_close[validation_split_ind:]


    dtrain_max = xgb.DMatrix(X_train, label=y_max_train, feature_names=feature_names)
    dval_max = xgb.DMatrix(X_val, label=y_max_val, feature_names=feature_names)

    dtrain_min = xgb.DMatrix(X_train, label=y_min_train, feature_names=feature_names)
    dval_min = xgb.DMatrix(X_val, label=y_min_val, feature_names=feature_names)

    dtrain_close = xgb.DMatrix(X_train, label=y_close_train, feature_names=feature_names)
    dval_close = xgb.DMatrix(X_val, label=y_close_val, feature_names=feature_names)

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

    watchlist_max = [(dtrain_max, 'train_max'), (dval_max, 'val_max')]
    watchlist_min = [(dtrain_min, 'train_min'), (dval_min, 'val_min')]
    watchlist_close = [(dtrain_close, 'train_close'), (dval_close, 'val_close')]

    num_round = 10000
    early_stopping_rounds = 100

    bst_max = xgb.train(params=params, dtrain=dtrain_max, num_boost_round=num_round,
                        evals=watchlist_max, early_stopping_rounds=early_stopping_rounds)

    bst_min = xgb.train(params=params, dtrain=dtrain_min, num_boost_round=num_round,
                        evals=watchlist_min, early_stopping_rounds=early_stopping_rounds)

    bst_close = xgb.train(params=params, dtrain=dtrain_close, num_boost_round=num_round,
                          evals=watchlist_close, early_stopping_rounds=early_stopping_rounds)

    max_best_ntree_limit = bst_max.best_ntree_limit
    min_best_ntree_limit = bst_min.best_ntree_limit
    close_best_ntree_limit = bst_close.best_ntree_limit

    return _get_predictor_func(bst_max, bst_min, bst_close,
                               max_best_ntree_limit, min_best_ntree_limit, close_best_ntree_limit)


def _get_predictor_func(bst_max, bst_min, bst_close,
                        max_best_ntree_limit, min_best_ntree_limit, close_best_ntree_limit):
    def _predictor_func(feature_values):
        x = np.array([feature_values])
        dmatrix = xgb.DMatrix(x)
        try:
            max_preds = bst_max.predict(dmatrix, n_tree_limit=max_best_ntree_limit)
            min_preds = bst_min.predict(dmatrix, n_tree_limit=min_best_ntree_limit)
            close_preds = bst_close.predict(dmatrix, n_tree_limit=close_best_ntree_limit)
        except XGBoostError:
            max_preds = bst_max.predict(dmatrix)
            min_preds = bst_min.predict(dmatrix)
            close_preds = bst_close.predict(dmatrix)

        assert len(max_preds) == 1 and len(min_preds) == 1 and len(close_preds) == 1
        return max_preds[0], min_preds[0], close_preds[0]
    return _predictor_func
