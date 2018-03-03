from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import DFDataset
from lambdatrader.signals.data_analysis.df_features import DFFeatureSet
from lambdatrader.signals.data_analysis.df_values import CloseReturn
from lambdatrader.signals.data_analysis.factories import FeatureSets
from lambdatrader.utilities.utils import seconds

store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

symbol = 'BTC_ETH'

num_days = 500
symbol_end = store.get_pair_period_newest_date(symbol)
start_date = symbol_end - seconds(days=num_days)

f = FeatureSets

feature_set = f.get_feature_set_1()
value_set = DFFeatureSet(features=[CloseReturn(n_candles=48, period=M5)])

ds = DFDataset.compute(symbol, start_date=start_date, feature_set=feature_set, value_set=value_set)

feature_df = ds.feature_df
f_names = ds.feature_names

print(len(f_names))
