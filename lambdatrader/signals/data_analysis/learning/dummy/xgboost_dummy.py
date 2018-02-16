from operator import itemgetter

import sklearn.metrics as sklearn_metrics
import xgboost as xgb

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set,
)
from lambdatrader.signals.data_analysis.values import (
    make_cont_trade_return,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_CVC'

day_offset = 7

# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*30)
dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24)
# dataset_start_date = latest_market_date - seconds(days=day_offset, minutes=30)

dataset_end_date = latest_market_date - seconds(days=day_offset)

dataset_len = dataset_end_date - dataset_start_date

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_small_feature_func_set()),
                                           value_function=make_cont_trade_return(),
                                           cache_and_get_cached=True,
                                           value_function_key='trade_return_15_mins')

print('created/loaded dataset\n')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()

train_ratio = 0.7

n_samples = len(y)
split_ind = int(train_ratio * n_samples)

X_train = X[:split_ind]
y_train = y[:split_ind]

X_test = X[split_ind:]
y_test = y[split_ind:]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print('xtrain', X_train)
# print('xtest', X_test)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

param = {'max_depth': 2, 'eta': 1, 'silent': 1}

num_round = 1000  # the number of training iterations

bst = xgb.train(param, dtrain, num_boost_round=num_round)
print(bst)

pred = bst.predict(dtest)

mse = sklearn_metrics.mean_squared_error(y_test, pred)

pred_real = list(zip(pred, y_test))

for p, y in zip(pred, y_test):
    # print('{:.6f}, {:.6f}'.format(p, y))
    # print(p>0, y>0)
    pass

sorted_by_pred = list(reversed(sorted(pred_real, key=itemgetter(0))))
sorted_by_real = list(reversed(sorted(pred_real, key=itemgetter(1))))

print()
print('pred, real:')
for item1, item2 in zip(sorted_by_pred, sorted_by_real):
    # print('{:30}'.format('{:.6f}, {:.6f}'.format(*item1)))
    print('{:30}{:30}'.format('{:.6f}, {:.6f}'.format(*item1), '{:.6f}, {:.6f}'.format(*item2)))

print()
print('mse:', mse)

print('len', len(pred))

true_pos_count = sum([1 for pred, real in pred_real if pred >= 0.03 and real >= 0.03])
false_pos_count = sum([1 for pred, real in pred_real if pred >= 0.03 and real < 0.03])

pred_pos_count = true_pos_count + false_pos_count

real_pos_count = sum([1 for _, real in pred_real if real >= 0.03])
real_pos_ratio = real_pos_count / len(pred)

print('real pos count-ratio', real_pos_count, real_pos_ratio)

if pred_pos_count == 0:
    print('no positive prediction')
    exit(0)

precision = true_pos_count / (true_pos_count + false_pos_count)
recall = true_pos_count / real_pos_count

print()
print('true-false pos count', true_pos_count, false_pos_count)

print()
print('precision', precision)
print('recall', recall)

true_pos_total = sum([real for pred, real in pred_real if pred >= 0.03 and real >= 0.03])
false_pos_total = sum([real for pred, real in pred_real if pred >= 0.03 and real < 0.03])

true_pos_avg = true_pos_total / true_pos_count
false_pos_avg = false_pos_total / false_pos_count

total = true_pos_total + false_pos_total

print()
print('total', total)
print('true-false pos total', true_pos_total, false_pos_total)


print()
print('true-false pos avg', true_pos_avg, false_pos_avg)
