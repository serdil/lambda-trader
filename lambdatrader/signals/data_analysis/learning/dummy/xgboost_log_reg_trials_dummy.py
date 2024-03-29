from operator import itemgetter

import xgboost as xgb
from xgboost.core import XGBoostError

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set,
)
from lambdatrader.signals.data_analysis.values import (
    make_cont_max_price_in_future,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    from lambdatrader.shelve_cache import shelve_cache_save
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

day_offset = 7

# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*500)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*365)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*200)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*120)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*90)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*60)
dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*30)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24)
# dataset_start_date = latest_market_date - seconds(days=day_offset, minutes=30)

dataset_end_date = latest_market_date - seconds(days=day_offset)

dataset_len = dataset_end_date - dataset_start_date


dataset_symbol = 'BTC_GAME'

num_candles = 12
value_func = make_cont_max_price_in_future(num_candles=num_candles, candle_period=M5)
value_func_name = 'cont_max_price_{}'.format(num_candles)

feature_functions = list(get_small_feature_func_set())
feature_funcs_name = 'small'


dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=feature_functions,
                                           value_function=value_func,
                                           cache_and_get_cached=True,
                                           feature_functions_key=feature_funcs_name,
                                           value_function_key=value_func_name)

print('created/loaded dataset\n')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()

train_ratio = 0.8
gap = num_candles

n_samples = len(y)

test_split = int(train_ratio * n_samples)

X_train = X[:test_split-gap]
y_train = y[:test_split-gap]

print(X_train)
print(y_train)

X_test = X[test_split:]
y_test = y[test_split:]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

params = {
    'silent': 1,
    'booster': 'gblinear',

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

watchlist = [(dtrain, 'train'), (dtest, 'test')]
num_round = 2000
early_stopping_rounds = 20

bst = xgb.train(params=params,
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds)

feature_importances = bst.get_fscore()

print()
print('feature importances:')
for f_name, imp in list(reversed(sorted(feature_importances.items(), key=itemgetter(1))))[:20]:
    print(f_name, ':', imp)

best_ntree_limit = bst.best_ntree_limit

try:
    pred = bst.predict(dtest, ntree_limit=best_ntree_limit)
except XGBoostError:
    pred = bst.predict(dtest)

pred_real = list(zip(pred, y_test))

reverse_sorted_by_pred = list(sorted(pred_real, key=lambda x: (x[0],x[1])))
reverse_sorted_by_real = list(sorted(pred_real, key=lambda x: (x[1],x[0])))

sorted_by_pred = list(reversed(sorted(pred_real, key=lambda x: (x[0],x[1]))))
sorted_by_real = list(reversed(sorted(pred_real, key=lambda x: (x[1],x[0]))))


print()
print('+++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST++++')

print()
print('sorted pred, real:')
for item1, item2 in list(zip(sorted_by_pred, sorted_by_real))[:50]:
    print('{:30}{:30}'.format('{:.6f}, {:.6f}'.format(*item1), '{:.6f}, {:.6f}'.format(*item2)))

print()
print('reverse sorted pred, real:')
for item1, item2 in list(zip(reverse_sorted_by_pred, reverse_sorted_by_real))[:50]:
    print('{:30}{:30}'.format('{:.6f}, {:.6f}'.format(*item1), '{:.6f}, {:.6f}'.format(*item2)))

print()
print('unsorted pred, real')
for pred, real in pred_real[-100:]:
    print('{:.6f}, {:.6f}'.format(pred, real))

#
# print()
# for change_level in np.arange(0.01, 0.51, 0.01):
#     num_pos = sum([1 for pred, real in pred_real if pred >= change_level])
#     num_true_pos = sum([1 for pred, real in pred_real if pred >= change_level and real >= change_level])
#     num_false_pos = sum([1 for pred, real in pred_real if pred >= change_level and real < change_level])
#     num_failing = sum([1 for pred, real in pred_real if pred >= change_level and real < 0.005])
#     print('level:{:.2} pos:{} true_pos:{} false_pos:{} failing:{}'.format(change_level, num_pos, num_true_pos, num_false_pos, num_failing))

# xgb.plot_importance(bst)
# xgb.plot_tree(bst)
# pyplot.show()
