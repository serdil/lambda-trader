from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.datasets import create_pair_dataset_from_history
from lambdatrader.signals.data_analysis.feature_sets import get_large_feature_func_set
from lambdatrader.signals.data_analysis.values import (
    make_cont_max_price_in_fifteen_mins,
)
from lambdatrader.utilities.utils import get_now_timestamp, date_floor, seconds

market_info = BacktestingMarketInfo(candlestick_store=
                                    ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

print(market_info)

now_date = date_floor(get_now_timestamp())

dataset_symbol = 'BTC_ETH'
dataset_start_date = now_date - seconds(days=7, hours=24)
dataset_end_date = now_date - seconds(days=7)

dataset = create_pair_dataset_from_history(market_info=market_info,
                                           pair=dataset_symbol,
                                           start_date=dataset_start_date,
                                           end_date=dataset_end_date,
                                           feature_functions=list(get_large_feature_func_set()),
                                           value_function=make_cont_max_price_in_fifteen_mins())

print(dataset)
print('number of data points:', len(dataset.get_numpy_feature_matrix()))
print('number of features:', len(dataset.get_first_feature_names()))
print([data_point.value for data_point in dataset.data_points])
