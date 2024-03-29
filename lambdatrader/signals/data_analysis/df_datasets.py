import hashlib
import os
import time

import pandas as pd
import xgboost as xgb
from pandas.core.base import DataError
from sklearn.datasets import dump_svmlight_file

from lambdatrader.candlestick_stores.sqlitestore import SQLiteCandlestickStore
from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.exchanges.enums import POLONIEX, ExchangeEnum
from lambdatrader.signals.data_analysis.df_features import LookbackFeature
from lambdatrader.signals.data_analysis.utils import date_str_to_timestamp
from lambdatrader.utilities.utils import get_project_directory


class DatasetDescriptor:
    def __init__(self, pairs, feature_set, value_set, start_date, end_date, exchanges=(POLONIEX,),
                 interleaved=False):

        if pairs is None:
            pairs = SQLiteCandlestickStore.get_for_exchange(POLONIEX).get_pairs()

        if isinstance(pairs, str):
            pairs = [pairs]

        if isinstance(exchanges, ExchangeEnum):
            exchanges = [exchanges]

        if len(exchanges) > 1:
            raise NotImplementedError('Multiple exchanges not implemented.')

        if not interleaved and len(pairs) > 1:
            raise ValueError('There should be exactly 1 pair if interleaved is False.')

        if not interleaved and len(exchanges) > 1:
            raise ValueError('There should be exactly 1 exchange if interleaved is False.')

        self.pairs = sorted(pairs)
        self.feature_set = feature_set
        self.value_set = value_set
        self.start_date = start_date
        self.end_date = end_date
        self.exchanges = exchanges
        self.interleaved = interleaved

    @property
    def descriptive_name(self):
        pair_names = ','.join(self.pairs)
        feature_names = ','.join(self.feature_set.feature_names)
        value_names = ','.join(self.value_set.feature_names)
        exchange_names = ','.join([e.name for e in self.exchanges])
        start_date = self.start_date
        end_date = self.end_date
        name = 'dataset_{}_{}_{}_{}_{}_{}'.format(pair_names, feature_names,
                                                  value_names, start_date, end_date, exchange_names)

        return name

    @property
    def hash(self):
        return hashlib.md5(self.descriptive_name.encode('utf-8')).hexdigest()

    @property
    def feature_names(self):
        return self.feature_set.feature_names

    @property
    def value_names(self):
        return self.value_set.feature_names

    @property
    def first_value_name(self):
        return self.value_names[0]

    def get_feature_set(self):
        return self.feature_set

    def get_value_set(self):
        return self.value_set


class SingleValueDatasetDescriptor(DatasetDescriptor):
    def __init__(self, pairs, feature_set, value_set, start_date, end_date,
                 exchanges=(POLONIEX,), interleaved=False):
        if len(value_set.features) > 1:
            raise ValueError('There should be one and only one value in the ValueSet.')
        super().__init__(pairs, feature_set, value_set,
                         start_date, end_date, exchanges, interleaved)

    @property
    def value(self):
        return self.value_set.features[0]


class SplitDatasetDescriptor:

    def __init__(self, train_descriptor, val_descriptor, test_descriptor):
        self.training = train_descriptor
        self.validation = val_descriptor
        self.test = test_descriptor

    @classmethod
    def create_with_train_val_test_date_ranges(cls, pairs, feature_set, value_set, split_date_range,
                                               exchanges=(POLONIEX,), interleaved=False):
        return cls._create_with_desc_class(pairs, feature_set, value_set, split_date_range,
                                           exchanges, interleaved, desc_class=DatasetDescriptor)

    @classmethod
    def create_single_value_with_train_val_test_date_ranges(cls, pairs, feature_set,
                                                            value_set, split_date_range,
                                                            exchanges=(POLONIEX,),
                                                            interleaved=False):
        return cls._create_with_desc_class(pairs, feature_set, value_set, split_date_range,
                                           exchanges, interleaved,
                                           desc_class=SingleValueDatasetDescriptor)

    @classmethod
    def _create_with_desc_class(cls, pairs, feature_set, value_set, split_date_range,
                                exchanges, interleaved, desc_class=DatasetDescriptor):
        train_dr = split_date_range.training
        val_dr = split_date_range.validation
        test_dr = split_date_range.test
        train_s, train_e = train_dr.start, train_dr.end
        val_s, val_e = val_dr.start, val_dr.end
        test_s, test_e = test_dr.start, test_dr.end

        train_dd = desc_class(pairs=pairs, feature_set=feature_set, value_set=value_set,
                              start_date=train_s, end_date=train_e, exchanges=exchanges,
                              interleaved=interleaved)

        val_dd = desc_class(pairs=pairs, feature_set=feature_set, value_set=value_set,
                            start_date=val_s, end_date=val_e, exchanges=exchanges,
                            interleaved=interleaved)

        test_dd = desc_class(pairs=pairs, feature_set=feature_set, value_set=value_set,
                             start_date=test_s, end_date=test_e, exchanges=exchanges,
                             interleaved=interleaved)

        return SplitDatasetDescriptor(train_dd, val_dd, test_dd)

    @property
    def first_value_name(self):
        return self.training.first_value_name


class DFDataset:

    def __init__(self, dfs, feature_df, value_df, feature_set, value_set,
                 feature_mapping, reverse_feature_mapping):
        self.dfs = dfs
        self.feature_df = feature_df
        self.value_df = value_df

        self.feature_set = feature_set,
        self.value_set = value_set

        self.feature_mapping = feature_mapping
        self.reverse_feature_mapping = reverse_feature_mapping

        self.return_values = []

    @classmethod
    def compute(cls, pair, feature_set, value_set, start_date=None,
                end_date=None, cs_store=None, normalize=True, error_on_missing=True):

        descriptor = DatasetDescriptor(pairs=[pair], feature_set=feature_set, value_set=value_set,
                                       start_date=start_date, end_date=end_date, exchanges=POLONIEX,
                                       interleaved=False)

        return cls.compute_from_descriptor(descriptor,
                                           normalize=normalize,
                                           error_on_missing=error_on_missing)

    @classmethod
    def compute_interleaved(cls, pairs, feature_set, value_set,
                            start_date=None, end_date=None, cs_store=None,
                            normalize=True, error_on_missing=True):

        descriptor = DatasetDescriptor(pairs=pairs, feature_set=feature_set, value_set=value_set,
                                       start_date=start_date, end_date=end_date, exchanges=POLONIEX,
                                       interleaved=True)

        return cls.compute_from_descriptor(descriptor,
                                           normalize=normalize,
                                           error_on_missing=error_on_missing)

    @classmethod
    def compute_from_descriptor(cls, descriptor, normalize=True, error_on_missing=True,
                                fillna=True):
        exchange = descriptor.exchanges[0]
        cs_store = SQLiteCandlestickStore.get_for_exchange(exchange)

        start_date = descriptor.start_date
        end_date = descriptor.end_date
        feature_set = descriptor.feature_set
        value_set = descriptor.value_set
        pairs = descriptor.pairs
        interleaved = descriptor.interleaved

        if interleaved:
            datasets = []
            for pair in pairs:
                try:
                    pair_desc = DatasetDescriptor(pairs=[pair],
                                                  feature_set=feature_set,
                                                  value_set=value_set,
                                                  start_date=start_date,
                                                  end_date=end_date,
                                                  exchanges=exchange,
                                                  interleaved=False)
                    ds = cls.compute_from_descriptor(pair_desc,
                                                     normalize=normalize,
                                                     error_on_missing=error_on_missing)
                    datasets.append(ds)
                except DataError as e:
                    if error_on_missing:
                        raise e
            return cls._interleave_datasets(datasets)
        else:
            pair = pairs[0]

            extended_start_date = start_date
            extended_end_date = end_date

            # TODO: fix normalization
            if extended_start_date is not None:
                extended_start_date = extended_start_date - feature_set.get_lookback()
            if extended_end_date is not None:
                extended_end_date = extended_end_date + value_set.get_lookforward()

            dfs = cs_store.get_agg_period_dfs(pair,
                                              start_date=extended_start_date,
                                              end_date=extended_end_date,
                                              periods=[M5, M15, H, H4, D],
                                              error_on_missing=error_on_missing)

            comp_start_time = time.time()

            feature_dfs = []
            feature_mapping = {}
            reverse_feature_mapping = {}

            # print('start_date', pd.Timestamp(start_date, unit='s'))
            # print('end_date', pd.Timestamp(end_date, unit='s'))
            # print('extended_start_date', pd.Timestamp(extended_start_date, unit='s'))
            # print('extended_end_date', pd.Timestamp(extended_end_date, unit='s'))

            for feature in feature_set.features:
                feature_start_date = start_date
                feature_end_date = end_date
                if feature_start_date is not None and isinstance(feature, LookbackFeature):
                    # print('feature lookback', feature.lookback)
                    feature_start_date = feature_start_date - feature.lookback

                # print('feature_start_date', pd.Timestamp(feature_start_date, unit='s'))
                # print('feature_end_date', pd.Timestamp(feature_end_date, unit='s'))

                dfs_for_feature = cls._get_df_slices(dfs, feature_start_date, feature_end_date)
                feature_df = feature.compute(dfs_for_feature)
                # print('feature_df start end', feature_df.index[0], feature_df.index[-1])
                feature_dfs.append(feature_df)
                for col_name in feature_df.columns.values:
                    feature_mapping[col_name] = feature
                reverse_feature_mapping[feature] = list(feature_df.columns.values)

            # feature_dfs = [f.compute(dfs) for f in feature_set.features]

            value_dfs = [v.compute(dfs) for v in value_set.features]

            feature_df = feature_dfs[0].join(feature_dfs[1:], how='inner')
            value_df = value_dfs[0].join(value_dfs[1:], how='inner')

            # print('final feature_df start end', feature_df.index[0], feature_df.index[-1])
            # all_na_cols = feature_df.columns[feature_df.isna().all()].tolist()
            # print(feature_df[all_na_cols])
            # print(all_na_cols)
            # print('end')

            if normalize:
                feature_df = feature_df.dropna()
                value_df = value_df.reindex(feature_df.index)

            if fillna:
                feature_df.fillna(0, inplace=True)

            # print('final feature_df start end after norm', feature_df.index[0],
            #       feature_df.index[-1])


            start_date_timestamp = pd.Timestamp(start_date, unit='s')
            end_date_timestamp = pd.Timestamp(end_date, unit='s')

            feature_df = feature_df.loc[start_date_timestamp:end_date_timestamp]
            value_df = value_df.loc[start_date_timestamp:end_date_timestamp]

            # print('dataset comp time: {:.3f}s'.format(time.time() - comp_start_time))

            return DFDataset(dfs, feature_df, value_df, feature_set, value_set,
                             feature_mapping, reverse_feature_mapping)

    @classmethod
    def _get_df_slices(cls, dfs, start_date, end_date):
        new_dfs = {}
        for key, df in dfs.items():
            new_dfs[key] = cls._get_df_slice(df, start_date, end_date)
        return new_dfs

    @classmethod
    def _get_df_slice(cls, df, start_date, end_date):
        if start_date is None:
            slice_start = None
        else:
            slice_start = pd.Timestamp(start_date, unit='s')

        if end_date is None:
            slice_end = None
        else:
            slice_end = pd.Timestamp(end_date, unit='s')

        date_slice = slice(slice_start, slice_end)
        return df.loc[date_slice]

    @classmethod
    def _interleave_datasets(cls, datasets):
        feature_mapping = datasets[0].feature_mapping
        reverse_feature_mapping = datasets[0].reverse_feature_mapping
        feature_dfs = [ds.feature_df for ds in datasets]
        value_dfs = [ds.value_df for ds in datasets]
        feature_df = pd.concat(feature_dfs).sort_index()
        value_df = pd.concat(value_dfs).sort_index()
        return DFDataset(dfs=None, feature_df=feature_df, value_df=value_df,
                         feature_set=None, value_set=None, feature_mapping=feature_mapping,
                         reverse_feature_mapping=reverse_feature_mapping)

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

    def add_feature_mapping(self):
        self.return_values.append(self.feature_mapping)
        return self

    def add_reverse_feature_mapping(self):
        self.return_values.append(self.reverse_feature_mapping)
        return self

    def get(self):
        return_values = tuple(self.return_values)
        self.return_values = []
        return return_values


LIBSVM_BATCH_SIZE = 10000


class XGBDMatrixDataset:

    def __init__(self, descriptor, dmatrix,
                 feature_names, feature_mapping, reverse_feature_mapping):
        self.descriptor = descriptor
        self.dmatrix = dmatrix

        self.feature_names = feature_names
        self.feature_mapping = feature_mapping
        self.reverse_feature_mapping = reverse_feature_mapping

    @classmethod
    def save_buffer(cls, descriptor: SingleValueDatasetDescriptor,
                    normalize=True, error_on_missing=True):
        dmatrix = (cls
                   .compute(descriptor, normalize=normalize, error_on_missing=error_on_missing)
                   .dmatrix)
        dmatrix.save_binary(cls._get_buffer_file_path(descriptor))

    @classmethod
    def save_libsvm(cls, descriptor: SingleValueDatasetDescriptor,
                    normalize=True, error_on_missing=True, batch_size=LIBSVM_BATCH_SIZE):
        batch_seconds = batch_size * M5.seconds()
        start_date = descriptor.start_date
        end_date = descriptor.end_date

        value_name = descriptor.value_set.features[0].name

        with open(cls._get_libsvm_file_path(descriptor), 'wb') as f:
            for batch_start_date in range(start_date, end_date, batch_size * M5.seconds()):
                print('computing batch...')
                batch_end_date = min(end_date, batch_start_date + batch_seconds)
                batch_descriptor = DatasetDescriptor(pairs=descriptor.pairs,
                                                     feature_set=descriptor.feature_set,
                                                     value_set=descriptor.value_set,
                                                     start_date=batch_start_date,
                                                     end_date=batch_end_date,
                                                     exchanges=descriptor.exchanges,
                                                     interleaved=descriptor.interleaved)
                x, y = (DFDataset.compute_from_descriptor(descriptor=batch_descriptor,
                                                          normalize=normalize,
                                                          error_on_missing=error_on_missing)
                        .add_feature_values()
                        .add_value_values(value_name=value_name)
                        .get())

                print('saving batch...')
                if len(x) > 0:
                    dump_svmlight_file(X=x, y=y, f=f, zero_based=True)

    @classmethod
    def _get_libsvm_file_path(cls, descriptor):
        return os.path.join(cls._dir_name(), descriptor.hash + '.txt')

    @classmethod
    def _get_buffer_file_path(cls, descriptor):
        return os.path.join(cls._dir_name(), descriptor.hash + '.buffer')

    @classmethod
    def _get_cache_path(cls, descriptor):
        return os.path.join(cls._dir_name(), descriptor.hash + '.cache')

    @staticmethod
    def _dir_name():
        path = os.path.join(get_project_directory(), 'data', 'libsvm')
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    @classmethod
    def compute(cls, descriptor: SingleValueDatasetDescriptor,
                normalize=True, error_on_missing=True):
        value_name = descriptor.value_set.features[0].name
        (x, y,
         feature_names,
         feature_mapping,
         reverse_feature_mapping) = (DFDataset
                                     .compute_from_descriptor(descriptor=descriptor,
                                                              normalize=normalize,
                                                              error_on_missing=error_on_missing)
                                     .add_feature_values()
                                     .add_value_values(value_name=value_name)
                                     .add_feature_names()
                                     .add_feature_mapping()
                                     .add_reverse_feature_mapping()
                                     .get())

        dmatrix = xgb.DMatrix(data=x, label=y, feature_names=feature_names)
        return XGBDMatrixDataset(descriptor=descriptor, dmatrix=dmatrix,
                                 feature_names=feature_names,
                                 feature_mapping=feature_mapping,
                                 reverse_feature_mapping=reverse_feature_mapping)

    @classmethod
    def load_libsvm_cached(cls, descriptor: SingleValueDatasetDescriptor):
        data_and_cache = '{}#{}'.format(cls._get_libsvm_file_path(descriptor),
                                        cls._get_cache_path(descriptor))
        dmatrix = xgb.DMatrix(data_and_cache)
        return XGBDMatrixDataset(descriptor, dmatrix,
                                 feature_mapping=None,
                                 reverse_feature_mapping=None)

    @classmethod
    def load_buffer(cls, descriptor: SingleValueDatasetDescriptor):
        buffer_path = cls._get_buffer_file_path(descriptor)
        dmatrix = xgb.DMatrix(buffer_path, feature_names=descriptor.feature_names)
        return XGBDMatrixDataset(descriptor, dmatrix,
                                 feature_mapping=None,
                                 reverse_feature_mapping=None)


class DateRange:

    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    @classmethod
    def from_str(cls, start_str=None, end_str=None):
        return DateRange(start=cls._parse_str(start_str), end=cls._parse_str(end_str))

    @staticmethod
    def _parse_str(date_str):
        if date_str is None:
            return None
        return date_str_to_timestamp(date_str)


class SplitDateRange:

    def __init__(self, train_dr, val_dr, test_dr):
        self.training = train_dr
        self.validation = val_dr
        self.test = test_dr
