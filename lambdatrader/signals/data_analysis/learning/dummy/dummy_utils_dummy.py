from datetime import datetime

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import MaxReturn, MinReturn, CloseReturn
from lambdatrader.signals.data_analysis.factories import DFFeatureSetFactory
from lambdatrader.utilities.utils import seconds

fsf = DFFeatureSetFactory


def get_x_and_y_close_max_min(symbol='BTC_ETH', day_offset=120, days=500, feature_set=fsf.get_small()):

    market_info = BacktestingMarketInfo(candlestick_store=
                                        ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

    latest_market_date = market_info.get_max_pair_end_time()

    dataset_start_date = latest_market_date - seconds(days=day_offset + days)

    dataset_end_date = latest_market_date - seconds(days=day_offset)

    print('start_date: {} end_date: {}'.format(datetime.utcfromtimestamp(dataset_start_date),
                                               datetime.utcfromtimestamp(dataset_end_date)))
    print()

    num_candles = 48

    max_return_v = MaxReturn(num_candles)
    min_return_v = MinReturn(num_candles)
    close_return_v = CloseReturn(num_candles)

    value_set = DFFeatureSet(features=[MaxReturn(num_candles),
                                       MinReturn(num_candles),
                                       CloseReturn(num_candles)])

    ds = DFDataset.compute(pair=symbol,
                           feature_set=feature_set,
                           value_set=value_set,
                           start_date=dataset_start_date,
                           end_date=dataset_end_date)

    X = ds.feature_values
    y_close = ds.get_value_values(close_return_v.name)
    y_max = ds.get_value_values(max_return_v.name)
    y_min = ds.get_value_values(min_return_v.name)

    return X, y_close, y_max, y_min
