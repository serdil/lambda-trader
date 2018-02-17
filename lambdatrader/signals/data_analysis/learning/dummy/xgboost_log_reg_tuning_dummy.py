from operator import itemgetter

import xgboost as xgb

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set_with_indicators,
)
from lambdatrader.signals.data_analysis.values import (
    make_binary_max_price_in_future,
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

dataset_symbol = 'BTC_LTC'

day_offset = 60

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

increase = 0.03
num_candles = 48
value_func = make_binary_max_price_in_future(increase=increase, num_candles=num_candles)
value_func_name = 'binary_max_price_{}_{}'.format(increase, num_candles)



dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_small_feature_func_set_with_indicators()),
                                           value_function=value_func,
                                           cache_and_get_cached=True,
                                           value_function_key=value_func_name)

print('created/loaded dataset\n')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()

train_ratio = 0.8
val_ratio = 0.9

n_samples = len(y)
val_split = int(train_ratio * n_samples)
test_split = int(val_ratio * n_samples)

X_train = X[:val_split]
y_train = y[:val_split]

X_val = X[val_split:test_split]
y_val = y[val_split:test_split]

X_test = X[test_split:]
y_test = y[test_split:]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)


params = {'max_depth': 6, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
watchlist = [(dtrain, 'train'), (dtest, 'test'), (dval, 'val')]
num_round = 10000
early_stopping_rounds = 100

bst = xgb.train(params=params,
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds)

feature_importances = bst.get_fscore()

print()
print('feature importances:')
for f_name, imp in reversed(sorted(feature_importances.items(), key=itemgetter(1))):
    print(f_name, ':', imp)

best_ntree_limit = bst.best_ntree_limit

pred = bst.predict(dval, ntree_limit=best_ntree_limit)

pred_real = list(zip(pred, y_val))

sorted_by_pred = list(reversed(sorted(pred_real, key=lambda x: (x[0],x[1]))))
sorted_by_real = list(reversed(sorted(pred_real, key=lambda x: (x[1],x[0]))))

print()
print('====VALIDATION=======VALIDATION=======VALIDATION=======VALIDATION=======VALIDATION===')

print()
print('pred, real:')
for item1, item2 in list(zip(sorted_by_pred, sorted_by_real))[:300]:
    print('{:30}{:30}'.format('{:.6f}, {:.6f}'.format(*item1), '{:.6f}, {:.6f}'.format(*item2)))

pred = bst.predict(dtest, ntree_limit=best_ntree_limit)

pred_real = list(zip(pred, y_test))

sorted_by_pred = list(reversed(sorted(pred_real, key=lambda x: (x[0],x[1]))))
sorted_by_real = list(reversed(sorted(pred_real, key=lambda x: (x[1],x[0]))))

print()
print('+++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST+++++++TEST++++')

print()
print('pred, real:')
for item1, item2 in list(zip(sorted_by_pred, sorted_by_real))[:300]:
    print('{:30}{:30}'.format('{:.6f}, {:.6f}'.format(*item1), '{:.6f}, {:.6f}'.format(*item2)))

# xgb.plot_importance(bst)
# xgb.plot_tree(bst)
# pyplot.show()
