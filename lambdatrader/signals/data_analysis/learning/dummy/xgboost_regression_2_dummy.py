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
    make_cont_trade_return,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    CandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_VIA'

# dataset_start_date = latest_market_date - seconds(days=7, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24*30)
dataset_start_date = latest_market_date - seconds(days=7, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24)
# dataset_start_date = latest_market_date - seconds(days=7, minutes=30)

dataset_end_date = latest_market_date - seconds(days=7)

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

print('xtrain', X_train)
print('xtest', X_test)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

paramold = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 0,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset

param = {'max_depth': 2, 'eta': 1, 'silent': 1}

num_round = 100  # the number of training iterations

bst = xgb.train(param, dtrain, num_boost_round=num_round)
print(bst)

pred = bst.predict(dtest)

print(pred)
print(y_test)

for p, y in zip(pred, y_test):
    print('{:.6f}, {:.6f}'.format(p, y))
    print(p>0, y>0)
