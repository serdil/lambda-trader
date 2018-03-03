from collections import namedtuple
from datetime import datetime

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import MaxReturn, MinReturn, CloseReturn
from lambdatrader.signals.data_analysis.factories import FeatureSets
from lambdatrader.utilities.utils import seconds

fsf = FeatureSets

DatasetInfo = namedtuple('DatasetInfo', ['ds', 'feature_df', 'feature_names', 'value_df',
                                         'close_return_name', 'max_return_name', 'min_return_name',
                                         'x', 'y_close', 'y_max', 'y_min'])

def get_dataset_info(symbol='BTC_ETH', day_offset=120, days=500, feature_set=fsf.get_small(), num_candles=48):
    market_info = BacktestingMarketInfo(
        candlestick_store=ChunkCachingCandlestickStore.get_for_exchange(ExchangeEnum.POLONIEX))

    latest_market_date = market_info.get_max_pair_end_time()

    dataset_start_date = latest_market_date - seconds(days=day_offset + days)

    dataset_end_date = latest_market_date - seconds(days=day_offset)

    print('start_date: {} end_date: {}'.format(datetime.utcfromtimestamp(dataset_start_date),
                                               datetime.utcfromtimestamp(dataset_end_date)))
    print()

    max_return_v = MaxReturn(num_candles)
    min_return_v = MinReturn(num_candles)
    close_return_v = CloseReturn(num_candles)

    value_set = DFFeatureSet(features=[close_return_v, max_return_v, min_return_v])

    ds = DFDataset.compute(pair=symbol, feature_set=feature_set, value_set=value_set,
                           start_date=dataset_start_date, end_date=dataset_end_date)

    ds_feature_df, ds_value_df = ds.feature_df, ds.value_df

    X = ds_feature_df.values
    y_close = ds_value_df[close_return_v.name].values
    y_max = ds_value_df[max_return_v.name]
    y_min = ds_value_df[min_return_v.name]

    return DatasetInfo(ds=ds,
                       feature_df=ds_feature_df,
                       feature_names=ds.feature_names,
                       value_df=ds_value_df,
                       close_return_name=max_return_v.name,
                       max_return_name=max_return_v.name,
                       min_return_name=min_return_v.name,
                       x=X, y_close=y_close,
                       y_max=y_max,
                       y_min=y_min)

