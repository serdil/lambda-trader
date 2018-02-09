import pprint
from typing import List

import numpy as np

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo

class Feature:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value


class FeatureSet:
    def __init__(self, features: List[Feature]):
        self._feature_values = []
        self._feature_names = []
        self._feature_dict = {}
        for feature in features:
            self._feature_dict[feature.name] = len(self._feature_values)
            self.feature_names.append(feature.name)
            self._feature_values.append(feature.value)
        self.num_features = len(self._feature_values)

    @property
    def feature_values(self):
        return self._feature_values

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def feature_dict(self):
        return self._feature_dict

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._feature_values[self._feature_dict[item]]
        elif isinstance(item, int):
            return self._feature_values[item]

    def __len__(self):
        return self.num_features

    def __repr__(self):
        dict_with_values = {}
        for name, ind in self._feature_dict.items():
            dict_with_values[name] = self._feature_values[ind]
        return 'Feature(dict={})'.format(pprint.pformat(dict_with_values))


class DataPoint:
    def __init__(self, feature_set: FeatureSet, value):
        self.feature_set = feature_set
        self.value = value

    def __repr__(self):
        return 'DataPoint(value={}, features={})'.format(self.value,
                                                         pprint.pformat(self.feature_set))

    @property
    def feature_values(self):
        return self.feature_set.feature_values

    @property
    def feature_names(self):
        return self.feature_set.feature_names

    @property
    def feature_dict(self):
        return self.feature_set.feature_dict


class DataSet:
    def __init__(self, data_points: List[DataPoint]):
        self.data_points = data_points

    def __repr__(self):
        return 'DataSet(data_points={})'.format(pprint.pformat(self.data_points))

    def get_first_feature_names(self):
        return self.data_points[0].feature_names

    def get_numpy_feature_matrix(self):
        return np.array([data_point.feature_values for data_point in self.data_points])

    def get_numpy_value_array(self):
        return np.array([data_point.value for data_point in self.data_points])


def create_pair_dataset_from_history(market_info: BacktestingMarketInfo,
                                     pair,
                                     start_date,
                                     end_date,
                                     feature_functions,
                                     value_function,
                                     cache_and_get_cached=False,
                                     feature_functions_key=None,
                                     value_function_key=None):
    if cache_and_get_cached:
        if feature_functions_key is None:
            feature_functions_key = len(tuple(feature_functions))
        if value_function_key is None:
            value_function_key = 'v'
        cache_key = _pair_dataset_cache_key_without_funcs(pair,
                                                          start_date,
                                                          end_date,
                                                          feature_functions_key,
                                                          value_function_key)
        try:
            from lambdatrader.shelve_cache import shelve_cache_get
            return shelve_cache_get(cache_key)
        except KeyError:
            print('dataset cache miss')
            return _compute_and_cache_pair_dataset(market_info,
                                                   pair,
                                                   start_date,
                                                   end_date,
                                                   feature_functions,
                                                   value_function,
                                                   feature_functions_key,
                                                   value_function_key)
    else:
        return _compute_pair_dataset(market_info,
                                     pair,
                                     start_date,
                                     end_date,
                                     feature_functions,
                                     value_function)


def _compute_and_cache_pair_dataset(market_info,
                                    pair,
                                    start_date,
                                    end_date,
                                    feature_functions,
                                    value_function,
                                    feature_functions_key,
                                    value_function_key):
    data_set = _compute_pair_dataset(market_info,
                                     pair,
                                     start_date,
                                     end_date,
                                     feature_functions,
                                     value_function)
    cache_key = _pair_dataset_cache_key_without_funcs(pair,
                                                      start_date,
                                                      end_date,
                                                      feature_functions_key,
                                                      value_function_key)
    from lambdatrader.shelve_cache import shelve_cache_save
    shelve_cache_save(cache_key, data_set)
    return data_set


def _compute_pair_dataset(market_info,
                          pair,
                          start_date,
                          end_date,
                          feature_functions,
                          value_function):

    market_info.set_market_date(start_date)

    data_points = []

    while market_info.market_date <= end_date:
        feature_set = compute_feature_set(market_info, pair, feature_functions)
        value = value_function(market_info, pair)
        data_points.append(DataPoint(feature_set=feature_set, value=value))
        market_info.inc_market_date()

    return DataSet(data_points=data_points)


# TODO doesn't work properly right now
def _pair_dataset_cache_key(pair, start_date, end_date, feature_functions, value_function):
    return pair, start_date, end_date, tuple(feature_functions), value_function


def _pair_dataset_cache_key_without_funcs(pair, start_date, end_date,
                                          feature_functions_key, value_function_key):
    return pair, start_date, end_date, feature_functions_key, value_function_key


def compute_feature_set(market_info, pair, feature_functions):
    features = []
    for i, func in enumerate(feature_functions):
        new_features = func(market_info, pair)
        for feature in new_features:
            features.append(feature)
    return FeatureSet(features=features)
