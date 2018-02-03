import time
from collections import defaultdict
from operator import itemgetter
from pprint import pprint

import numpy as np
import xgboost as xgb

import sklearn.metrics as sklearn_metrics
from matplotlib import pyplot

from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from xgboost import XGBRegressor

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import (
    get_large_feature_func_set, get_small_feature_func_set, get_alt_small_feature_func_set,
    get_small_feature_func_set_with_indicators,
)
from lambdatrader.signals.data_analysis.learning.dummy.learning_utils_dummy import (
    train_and_test_model, print_model_metrics,
)
from lambdatrader.signals.data_analysis.values import (
    make_cont_trade_return, make_cont_close_price_in_future, make_cont_close_price_in_fifteen_mins,
    make_binary_max_price_in_future,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


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


dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

params = {'max_depth': 6, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 100
cv_fold = 5

cv_out = xgb.cv(params=params,
                dtrain=dtrain,
                num_boost_round=num_round)

train_mean = cv_out['train-error-mean']
train_std = cv_out['train-error-std']
test_mean = cv_out['test-error-mean']
test_std = cv_out['test-error-std']

print('          {:<25}{:<25}{:<25}{:<25}'.format('train-error-mean', 'train-error-std', 'test-error-mean', 'test-error-std'))
for i, (v1, v2, v3, v4) in enumerate(zip(train_mean, train_std, test_mean, test_std)):
    print('{:<10}{:<25}{:<25}{:<25}{:<25}'.format(i, v1, v2, v3, v4))
