import pprint
from collections import defaultdict
from operator import itemgetter

import numpy as np
import sklearn.metrics as sklearn_metrics
import xgboost as xgb

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import CandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_small_feature_func_set,
)
from lambdatrader.signals.data_analysis.values import (
    make_cont_close_price_in_fifteen_mins,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_CVC'

day_offset = 7

# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*30)
# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*7)
dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24)
# dataset_start_date = latest_market_date - seconds(days=day_offset, minutes=30)

dataset_end_date = latest_market_date - seconds(days=day_offset)

dataset_len = dataset_end_date - dataset_start_date

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_small_feature_func_set()),
                                           value_function=make_cont_close_price_in_fifteen_mins(),
                                           cache_and_get_cached=True,
                                           value_function_key='close_price_in_future_15_mins')

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

param = {'max_depth': 3, 'eta': 1, 'silent': 0}

num_round = 2000  # the number of training iterations

bst = xgb.train(param, dtrain, num_boost_round=num_round)
print(bst)

pred = bst.predict(dtest)

mse = sklearn_metrics.mean_squared_error(y_test, pred)

pred_real = list(zip(pred, y_test))

sorted_by_pred = list(reversed(sorted(pred_real, key=itemgetter(0))))
sorted_by_real = list(reversed(sorted(pred_real, key=itemgetter(1))))

print()
print('pred, real:')
for item1, item2 in zip(sorted_by_pred, sorted_by_real):
    # print('{:30}'.format('{:.6f}, {:.6f}'.format(*item1)))
    print('{:30}{:30}'.format('{:.6f}, {:.6f}'.format(*item1), '{:.6f}, {:.6f}'.format(*item2)))

print()
print('mse:', mse)

real = y_test

pred_real_sorted = list(reversed(sorted(list(zip(pred, real)), key=itemgetter(0))))

pred_sign = pred > 0
real_sign = y_test > 0

sign_equal = np.equal(real_sign, pred_sign)

n_samples = len(y_test)

real_positive_ratio = np.sum(real_sign) / n_samples
sign_equal_ratio = np.sum(sign_equal) / n_samples

pred_real_sign = list(zip(pred_sign, real_sign))
false_positive_count = sum([1 for elem in pred_real_sign if elem == (True, False)])
true_positive_count = sum([1 for elem in pred_real_sign if elem == (True, True)])
false_negative_count = sum([1 for elem in pred_real_sign if elem == (False, True)])
true_negative_count = sum([1 for elem in pred_real_sign if elem == (False, False)])

true_positive_ratio = true_positive_count / n_samples
false_positive_ratio = false_positive_count / n_samples
true_negative_ratio = true_negative_count / n_samples
false_negative_ratio = false_negative_count / n_samples

true_positive_total = sum([real for pred, real in pred_real_sorted if pred > 0 and real > 0])

false_positive_total = sum([real for pred, real in pred_real_sorted if pred > 0 and real <= 0])

if true_positive_count == 0:
    true_positive_count = 0.0001
if false_positive_count == 0:
    false_positive_count = 0.0001

true_positive_avg = true_positive_total / true_positive_count
false_positive_avg = false_positive_total / false_positive_count

positive_avg = (true_positive_total + false_positive_total) / (
true_positive_count + false_positive_count)

metrics_dict = {
            'pred': pred,
            'real': y_test,
            'pred_real_sorted': pred_real_sorted,
            'pred_sign': pred_sign,
            'real_sign': real_sign,
            'sign_equal_ratio': sign_equal_ratio,
            'real_positive_ratio': real_positive_ratio,
            'mse': mse,
            'true_positive_ratio': true_positive_ratio,
            'false_positive_ratio': false_positive_ratio,
            'true_negative_ratio': true_negative_ratio,
            'false_negative_ratio': false_negative_ratio,
            'true_positive_total': true_positive_total,
            'false_positive_total': false_positive_total,
            'true_positive_avg': true_positive_avg,
            'false_positive_avg': false_positive_avg,
            'positive_avg': positive_avg
        }

metrics = defaultdict(str)
metrics.update(metrics_dict)

if 'importance' in metrics:
    print('IMPORTANCES:')
    for name, importance in metrics['importance']:
        print('{:<80} {}'.format(name, importance))
    print()

pred_real = metrics['pred_real_sorted']
print('pred, real:', pprint.pformat(pred_real))

pred_sign = metrics['pred_sign']
real_sign = metrics['real_sign']

pred_real_sign = list(zip(pred_sign, real_sign))
print('pred_sign, real_sign:', pprint.pformat(pred_real_sign))

print()
print('training time:', metrics['training_time'])
print('mse:', metrics['mse'], 'score:', metrics['score'])

print()
print('real positive ratio:', metrics['real_positive_ratio'])
print('sign equal ratio:', metrics['sign_equal_ratio'])
print()
print('true positive ratio:', metrics['true_positive_ratio'])
print('false positive ratio:', metrics['false_positive_ratio'])
print()
print('true negative ratio:', metrics['true_negative_ratio'])
print('false negative ratio:', metrics['false_negative_ratio'])
print()
print('true positive total:', metrics['true_positive_total'])
print('false positive total:', metrics['false_positive_total'])
print()
print('true positive avg:', metrics['true_positive_avg'])
print('false positive avg:', metrics['false_positive_avg'])
print()
print('positive avg:', metrics['positive_avg'])
print()
