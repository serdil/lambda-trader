from pandas.core.base import DataError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.factories import FeatureSets
from lambdatrader.signals.data_analysis.learning.dummy.dummy_utils_dummy import (
    get_dataset_info,
)
from lambdatrader.signals.data_analysis.learning.dummy.np_interleave_util import (
    interleave_2d, interleave_1d,
)
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output

all_symbols = set(SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs())

# symbols = ['BTC_ETH']
# symbols = ['BTC_VIA', 'BTC_SC', 'BTC_ETH']
# symbols = ['BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_ETH']
# symbols = ['BTC_LTC', 'BTC_ETC', 'BTC_XMR', 'BTC_SYS', 'BTC_VIA', 'BTC_SC', 'BTC_ETH']
symbols = sorted(list(all_symbols))

num_candles = 48

day_offset = 120
days = 10

feature_set = FeatureSets.get_small()

ds_infos = []


for symbol in symbols:
    print('loading {}'.format(symbol))
    try:
        ds_info = get_dataset_info(symbol=symbol, day_offset=day_offset, days=days, feature_set=feature_set)
        ds_infos.append(ds_info)
    except DataError:
        print('DataError: {}'.format(symbol))

xs = [ds_info.x for ds_info in ds_infos]
y_closes = [ds_info.y_close for ds_info in ds_infos]
y_maxs = [ds_info.y_max for ds_info in ds_infos]
y_mins = [ds_info.y_min for ds_info in ds_infos]

X = interleave_2d(xs, num_rows=num_candles)
y_close = interleave_1d(y_closes, num_rows=num_candles)
y_max = interleave_1d(y_maxs, num_rows=num_candles)
y_min = interleave_1d(y_mins, num_rows=num_candles)

feature_names = ds_infos[0].feature_names

n_samples = len(X)
gap = num_candles * len(symbols)

test_ratio = 0.8

test_split_ind = int(n_samples * test_ratio)


X_train = X[:test_split_ind - gap]
y_max_train = y_max[:test_split_ind - gap]
y_min_train = y_min[:test_split_ind - gap]
y_close_train = y_close[:test_split_ind - gap]

X_test = X[test_split_ind:]
y_max_test = y_max[test_split_ind:]
y_min_test = y_min[test_split_ind:]
y_close_test = y_close[test_split_ind:]

print('test set size:', len(X_test))
print('test set size / 288:', len(X_test) / 288)

n_estimators = 10

kwargs = {
    'max_depth': 20
}

rf_close = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, verbose=True, **kwargs)
rf_max = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, verbose=True, **kwargs)
rf_min = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, verbose=True, **kwargs)

print('starting training')
rf_close.fit(X_train, y_close_train)
print('training complete')
rf_max.fit(X_train, y_max_train)
print('training complete')
rf_min.fit(X_train, y_min_train)
print('training complete')


pred_max = rf_max.predict(X_test)
pred_min = rf_min.predict(X_test)
pred_close = rf_close.predict(X_test)

mse = mean_squared_error(y_close_test, pred_close)
print('close mse:', mse)


pred_real_max = list(zip(pred_max, y_max_test))
pred_real_min = list(zip(pred_min, y_min_test))
pred_real_close = list(zip(pred_close, y_close_test))


analyze_output(pred_real_max, pred_real_min, pred_real_close)
