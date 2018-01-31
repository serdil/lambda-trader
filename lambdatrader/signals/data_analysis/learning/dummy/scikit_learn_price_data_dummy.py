from collections import OrderedDict
from operator import itemgetter

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.features import get_feature_funcs_iter
from lambdatrader.signals.data_analysis.values import make_cont_max_price_in_fifteen_mins
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

print(market_info)

latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_ETH'
dataset_start_date = latest_market_date - seconds(days=7, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=7, minutes=20)

dataset_end_date = latest_market_date - seconds(days=7)

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_feature_funcs_iter()),
                                           value_function=make_cont_max_price_in_fifteen_mins(),
                                           cache_and_get_cached=True)

print('created dataset')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()

# print(feature_names)
print(X)
print(y)

print('X shape:', X.shape)
print('y shape:', y.shape)

n_samples = len(y)

train_ratio = 0.7
split_ind = int(train_ratio * n_samples)

X_train = X[:split_ind]
y_train = y[:split_ind]

X_test = X[split_ind:]
y_test = y[split_ind:]


rfr = RandomForestRegressor(n_estimators=50)
rfr.fit(X_train, y_train)

rfr_pred = rfr.predict(X_test)

rfr_mse = metrics.mean_squared_error(y_test, rfr_pred)
rfr_score = rfr.score(X_test, y_test)

pred_sign = rfr_pred > 0
real_sign = y_test > 0

sign_equal = np.equal(real_sign, pred_sign)

# print('predictions:', rfr_pred * 100)
# print('real:', y_test * 100)

rfr_importance = rfr.feature_importances_
name_importance = zip(feature_names, rfr_importance)
name_importance_sorted = list(reversed(sorted(name_importance, key=itemgetter(1))))
importance_dict = OrderedDict(name_importance_sorted)

for name, importance in name_importance_sorted:
    if importance != 0:
        print(name, importance)

print('mse:', rfr_mse, 'score:', rfr_score)

unique, counts = np.unique(sign_equal, return_counts=True)
print(dict(zip(unique, counts)))

save('rfr', rfr)
