from sklearn.ensemble import RandomForestRegressor

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import get_large_feature_func_set
from lambdatrader.signals.data_analysis.learning.dummy.learning_utils_dummy import (
    train_and_test_model, print_model_metrics,
)
from lambdatrader.signals.data_analysis.values import make_cont_max_price_in_fifteen_mins
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)


market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_ETH'

# dataset_start_date = latest_market_date - seconds(days=7, hours=24*60)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24*30)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24)
dataset_start_date = latest_market_date - seconds(days=7, minutes=20)

dataset_end_date = latest_market_date - seconds(days=7)

dataset_len = dataset_end_date - dataset_start_date

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_large_feature_func_set()),
                                           value_function=make_cont_max_price_in_fifteen_mins(),
                                           cache_and_get_cached=True)

print('created/loaded dataset\n')

feature_names = dataset.get_first_feature_names()

X = dataset.get_numpy_feature_matrix()
y = dataset.get_numpy_value_array()

n_estimators = 100
rfr = RandomForestRegressor(n_estimators=n_estimators)

stats = train_and_test_model(dataset, rfr)
print_model_metrics(stats)

model_name = 'rfr_max_price_{}_{}'.format(dataset_len, n_estimators)

save(model_name, rfr)
print('saved model:', model_name)
