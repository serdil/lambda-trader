from xgboost import XGBClassifier

from xgboost import XGBClassifier

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.candlestickstore import ChunkCachingCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.shelve_cache import shelve_cache_save
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import get_large_feature_func_set
from lambdatrader.signals.data_analysis.learning.dummy.learning_utils_dummy import (
    train_and_test_model,
)
from lambdatrader.signals.data_analysis.values import (
    make_class_max_price_in_future,
)
from lambdatrader.utilities.utils import seconds


def cache_key(model_name):
    return 'dummy_model_{}'.format(model_name)


def save(model_name, model):
    shelve_cache_save(cache_key(model_name), model)

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))


latest_market_date = market_info.get_max_pair_end_time()

dataset_symbol = 'BTC_VIA'

# dataset_start_date = latest_market_date - seconds(days=7, hours=24*60)
dataset_start_date = latest_market_date - seconds(days=7, hours=24*30)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24*7)
# dataset_start_date = latest_market_date - seconds(days=7, hours=24)
# dataset_start_date = latest_market_date - seconds(days=7, minutes=30)

dataset_end_date = latest_market_date - seconds(days=7)

dataset_len = dataset_end_date - dataset_start_date

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_large_feature_func_set()),
                                           value_function=
                                           make_class_max_price_in_future(num_candles=12,
                                                                          candle_period=M5),
                                           cache_and_get_cached=True,
                                           value_function_key='max_price_class_60_min')

print('created/loaded dataset\n')

xgbc = XGBClassifier()

metrics = train_and_test_model(dataset, xgbc, train_ratio=0.7, classification_task=True)

for key, value in metrics.items():
    print(key, ':', value)

model_name = 'xgbc_max_price_class_60_min_{}'.format(dataset_len)

save(model_name, xgbc)
print('saved model:', model_name)
