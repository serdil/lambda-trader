import time

import pandas as pd
from pandas.core.base import DataError

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.exchanges.enums import POLONIEX


class DFDataset:

    def __init__(self, dfs, feature_df, value_df, feature_set, value_set):
        self.dfs = dfs
        self.feature_df = feature_df
        self.value_df = value_df

        self.feature_set = feature_set,
        self.value_set = value_set

        self.return_values = []

    @classmethod
    def compute(cls, pair, feature_set, value_set, start_date=None,
                end_date=None, cs_store=None, normalize=True, error_on_missing=True):
        if cs_store is None:
            cs_store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)

        start_date = start_date - feature_set.get_lookback()
        end_date = end_date + value_set.get_lookforward()

        dfs = cs_store.get_agg_period_dfs(pair,
                                          start_date=start_date,
                                          end_date=end_date,
                                          periods=[M5, M15, H, H4, D],
                                          error_on_missing=error_on_missing)

        start_time = time.time()
        feature_dfs = [f.compute(dfs) for f in feature_set.features]

        value_dfs = [v.compute(dfs) for v in value_set.features]

        feature_df = feature_dfs[0].join(feature_dfs[1:], how='inner')
        value_df = value_dfs[0].join(value_dfs[1:], how='inner')

        if normalize:
            feature_df = feature_df.dropna()
            value_df = value_df.reindex(feature_df.index)

        print('dataset comp time: {:.3f}s'.format(time.time() - start_time))

        return DFDataset(dfs, feature_df, value_df, feature_set, value_set)

    @classmethod
    def compute_interleaved(cls, pairs, feature_set, value_set,
                            start_date=None, end_date=None, cs_store=None,
                            normalize=True, error_on_missing=True):
        if cs_store is None:
            cs_store = SQLiteCandlestickStore.get_for_exchange(POLONIEX)
        if pairs is None:
            pairs = cs_store.get_pairs()

        datasets = []
        for pair in pairs:
            try:
                ds = cls.compute(pair=pair,
                        feature_set=feature_set,
                        value_set=value_set,
                        start_date=start_date,
                        end_date=end_date,
                        cs_store=cs_store,
                        normalize=normalize,
                        error_on_missing=error_on_missing)
                datasets.append(ds)
            except DataError as e:
                if error_on_missing:
                    raise e

        return cls._interleave_datasets(datasets)

    @classmethod
    def _interleave_datasets(cls, datasets):
        feature_dfs = [ds.feature_df for ds in datasets]
        value_dfs = [ds.value_df for ds in datasets]
        feature_df = pd.concat(feature_dfs).sort_index()
        value_df = pd.concat(value_dfs).sort_index()
        return DFDataset(dfs=None, feature_df=feature_df, value_df=value_df,
                         feature_set=None, value_set=None)

    @property
    def feature_names(self):
        return self.feature_df.columns.values.tolist()

    @property
    def value_names(self):
        return self.value_df.columns.values.tolist()

    def get_feature_values(self, start_date=None, end_date=None):
        if start_date or end_date:
            return self.feature_df.loc[start_date:end_date].values
        else:
            return self.feature_df.values

    def get_feature_row(self, date):
        return self.get_feature_values(start_date=date, end_date=date).reshape(1, -1)

    def get_value_values(self, value_name=None, start_date=None, end_date=None):
        if value_name is None:
            if start_date or end_date:
                return self.value_df.loc[start_date:end_date].values
            else:
                return self.value_df.values
        else:
            if start_date or end_date:
                return self.value_df[value_name].loc[start_date:end_date].values
            else:
                return self.value_df[value_name].values

    def get_value_row(self, date, value_name=None):
        return (self.get_value_values(value_name=value_name, start_date=date, end_date=date)
                .reshape(1, -1))

    def add_feature_names(self):
        self.return_values.append(self.feature_names)
        return self

    def add_value_names(self):
        self.return_values.append(self.value_names)
        return self

    def add_feature_df(self):
        self.return_values.append(self.feature_df)
        return self

    def add_value_df(self):
        self.return_values.append(self.value_df)
        return self

    def add_feature_values(self, start_date=None, end_date=None):
        self.return_values.append(self.get_feature_values(start_date, end_date))
        return self

    def add_value_values(self, value_name=None, start_date=None, end_date=None):
        self.return_values.append(self.get_value_values(value_name, start_date, end_date))
        return self

    def get(self):
        return_values = tuple(self.return_values)
        self.return_values = []
        return return_values

