from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.features import get_feature_funcs_iter
from lambdatrader.signals.data_analysis.values import make_cont_max_price_in_fifteen_mins
from lambdatrader.utilities.utils import seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

print(market_info)

latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_ETH'
dataset_start_date = latest_market_date - seconds(days=7, hours=24)
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

print(feature_names)
print(X)
print(y)

print('X shape:', X.shape)
print('y shape:', y.shape)

n_samples = len(y)

train_ratio = 0.7

X_train = X[:int(train_ratio * n_samples)]
y_train = y[:int(train_ratio * n_samples)]

X_test = X[int(train_ratio * n_samples):]
y_test = y[int(train_ratio * n_samples):]


rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

mse = metrics.mean_squared_error(y_test, predictions)
score = rf.score(X_test, y_test)

print('predictions:', predictions * 100)
print('real:', y_test * 100)

print('mse:', mse, 'score:', score)
