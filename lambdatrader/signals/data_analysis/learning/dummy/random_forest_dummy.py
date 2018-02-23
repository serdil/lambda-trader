from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.df_datasets import Dataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import MaxReturn, CloseReturn, MinReturn
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory
from lambdatrader.signals.data_analysis.learning.dummy.xgboost_analysis_utils_dummy import \
    analyze_output
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

day_offset = 120

# dataset_start_date = latest_market_date - seconds(days=day_offset, hours=24*1000)
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

print('start_date: {} end_date: {}'.format(datetime.utcfromtimestamp(dataset_start_date),
                                           datetime.utcfromtimestamp(dataset_end_date)))
print()


dataset_symbol = 'BTC_ETH'

fsf = DFFeatureSetFactory

num_candles = 48

feature_set = fsf.get_small()

max_return_v = MaxReturn(num_candles)
min_return_v = MinReturn(num_candles)
close_return_v = CloseReturn(num_candles)

value_set = DFFeatureSet(features=[MaxReturn(num_candles),
                                   MinReturn(num_candles),
                                   CloseReturn(num_candles)])

ds = Dataset.compute(pair=dataset_symbol,
                     feature_set=feature_set,
                     value_set=value_set,
                     start_date=dataset_start_date,
                     end_date=dataset_end_date)


# ================================= DATASET CREATION UP TILL HERE ==================================

feature_names = ds.feature_names


X_df = ds.feature_df.dropna()

y_max_s = ds.value_df[max_return_v.name].reindex(X_df.index)
y_min_s = ds.value_df[min_return_v.name].reindex(X_df.index)
y_close_s = ds.value_df[close_return_v.name].reindex(X_df.index)

X = X_df.values
y_max = y_max_s.values
y_min = y_min_s.values
y_close = y_close_s.values

train_ratio = 0.7
validation_ratio = 1.0
gap = num_candles

n_samples = len(X)

validation_split_ind = int(train_ratio * n_samples)
test_split_ind = int(validation_ratio * n_samples)

X_train = X[:validation_split_ind - gap]
y_max_train = y_max[:validation_split_ind - gap]
y_min_train = y_min[:validation_split_ind - gap]
y_close_train = y_close[:validation_split_ind - gap]


X_val = X[validation_split_ind:test_split_ind]
y_max_val = y_max[validation_split_ind:test_split_ind]
y_min_val = y_min[validation_split_ind:test_split_ind]
y_close_val = y_close[validation_split_ind:test_split_ind]


X_test = X[test_split_ind:]
y_max_test = y_max[test_split_ind:]
y_min_test = y_min[test_split_ind:]
y_close_test = y_close[test_split_ind:]

print('created/loaded dataset\n')

n_estimators = 1000

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


pred_max = rf_max.predict(X_val)
pred_min = rf_min.predict(X_val)
pred_close = rf_close.predict(X_val)

mse = mean_squared_error(y_close_val, pred_close)
print('close mse:', mse)


pred_real_max = list(zip(pred_max, y_max_val))
pred_real_min = list(zip(pred_min, y_min_val))
pred_real_close = list(zip(pred_close, y_close_val))


analyze_output(pred_real_max, pred_real_min, pred_real_close)



