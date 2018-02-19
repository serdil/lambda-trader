import time

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.exchanges.enums import POLONIEX


class Dataset:

    def __init__(self, dfs, feature_df, value_df):
        self.dfs = dfs
        self.feature_df = feature_df
        self.value_df = value_df

    @classmethod
    def compute(cls, pair, feature_set, value_set, start_date=None, end_date=None, cs_store=None):
        if cs_store is None:
            cs_store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

        dfs = cs_store.get_agg_period_dfs(pair,
                                          start_date=start_date,
                                          end_date=end_date,
                                          periods=[M5, M15, H, H4, D])

        start_time = time.time()
        feature_dfs = [f.compute(dfs) for f in feature_set.features]

        value_dfs = [v.compute(dfs) for v in value_set.features]

        feature_df = feature_dfs[0].join(feature_dfs[1:], how='inner')
        value_df = value_dfs[0].join(value_dfs[1:], how='inner')
        print('dataset comp time: {:.3f}s'.format(time.time() - start_time))

        return Dataset(dfs, feature_df, value_df)

    @property
    def feature_names(self):
        return self.feature_df.columns.values.tolist()

    @property
    def value_names(self):
        return self.value_df.columns.values.tolist()