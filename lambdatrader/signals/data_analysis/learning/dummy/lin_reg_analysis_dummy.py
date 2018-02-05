
# get max prices, min prices and close prices for 1 hour
# train model to predict max price in 1 hour
# train model to predict min price in 1 hour
# determine trading gains for different (max_price_threshold, min_price_threshold) pairs.

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


dataset_symbol = 'BTC_SYS'

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

train_ratio = 0.8
gap = num_candles

n_samples = len(X)

test_split_ind = int(train_ratio * n_samples)

X_train = X[:test_split_ind - gap]
y_max_train = y_max[:test_split_ind - gap]
y_min_train = y_min[:test_split_ind - gap]
y_close_train = y_close[:test_split_ind - gap]


X_test = X[test_split_ind:]
y_max_test = y_max[test_split_ind:]
y_min_test = y_min[test_split_ind:]
y_close_test = y_close[test_split_ind:]

print(X_test)
print(y_max_test)
print(y_min_test)
print(y_close_test)

dtrain_max = xgb.DMatrix(X_train, label=y_max_train, feature_names=feature_names)
dtest_max = xgb.DMatrix(X_test, label=y_max_test, feature_names=feature_names)

dtrain_min = xgb.DMatrix(X_train, label=y_min_train, feature_names=feature_names)
dtest_min = xgb.DMatrix(X_test, label=y_min_test, feature_names=feature_names)

dtrain_close = xgb.DMatrix(X_train, label=y_close_train, feature_names=feature_names)
dtest_close = xgb.DMatrix(X_test, label=y_close_test, feature_names=feature_names)

params = {
    'silent': 1,
    'booster': 'gbtree',

    'objective': 'reg:linear',
    'base_score': 0,
    'eval_metric': 'rmse',

    'eta': 0.01,
    'gamma': 0,
    'max_depth': 1,
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

watchlist_max = [(dtrain_max, 'train_max'), (dtest_max, 'test_max')]
watchlist_min = [(dtrain_min, 'train_min'), (dtest_min, 'test_min')]
watchlist_close = [(dtrain_close, 'train_close'), (dtest_close, 'test_close')]

num_round = 10000
early_stopping_rounds = 2000

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

pred_max = bst_max.predict(dtest_max, ntree_limit=max_best_ntree_limit)
pred_min = bst_min.predict(dtest_min, ntree_limit=min_best_ntree_limit)
pred_close = bst_close.predict(dtest_close, ntree_limit=close_best_ntree_limit)

pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))
pred_real_close = list(zip(pred_close, y_close_test))

pred_real_max_min_close = list(zip(pred_real_max, pred_real_min, pred_real_close))

sort_key_pred_max_min_close = lambda a: (a[0][0], a[1][0], a[2][0])
sort_key_pred_max_min_close_sum = lambda a: a[0][0] + a[1][0] + a[2][0]

sorted_normally = list(reversed(sorted(pred_real_max_min_close, key=sort_key_pred_max_min_close)))
sorted_by_sum = list(reversed(sorted(pred_real_max_min_close, key=sort_key_pred_max_min_close_sum)))

tp_level = 0.8
p_sum = 0
num_eval = 100
for pred_real_max, pred_real_min, pred_real_close in sorted_by_sum[:num_eval]:
    if pred_real_max[1] >= pred_real_max[0] * tp_level:
        profit = pred_real_max[0] * tp_level
    else:
        profit = pred_real_close[1]
    p_sum += profit
    print(pred_real_max, pred_real_min, pred_real_close, 'profit', profit)
avg_profit = p_sum / num_eval
print('p_sum', p_sum, 'avg', avg_profit)






# profits = []
# for pred_real_max, pred_real_min, pred_real_close in pred_real_max_min_close:
#     if pred_real_max[1] >= pred_real_max[0] * tp_level:
#         profit = pred_real_max[0] * tp_level
#     else:
#         profit = pred_real_close[1]
#     profits.append(profit)
#
# pred_real_max_min_close_profit = list(zip(pred_real_max_min_close, profits))
#
# sort_key_profit = lambda a: a[1]
#
# sorted_by_profit = list(reversed(sorted(pred_real_max_min_close_profit, key=sort_key_profit)))
#
# for (pred_real_max, pred_real_min, pred_real_close), profit in sorted_by_profit:
#     print(pred_real_max, pred_real_min, pred_real_close, 'profit', profit)
