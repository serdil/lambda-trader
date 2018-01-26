from typing import List

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo


class Feature:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value


class FeatureSet:
    def __init__(self, features: List(Feature)):
        self.feature_values = []
        self.feature_dict = {}
        for feature in features:
            self.feature_dict[feature.name] = len(self.feature_values)
            self.feature_values.append(feature.value)
        self.num_features = len(self.feature_values)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.feature_values[self.feature_dict[item]]
        elif isinstance(item, int):
            return self.feature_values[item]

    def __len__(self):
        return self.num_features


class DataPoint:
    def __init__(self, feature_set: FeatureSet, value):
        self.features = feature_set
        self.value = value


class DataSet:
    def __init__(self, data_points: List(DataPoint)):
        self.data_points = data_points


def create_pair_dataset_from_history(market_info: BacktestingMarketInfo,
                                     pair,
                                     start_date,
                                     end_date,
                                     feature_functions,
                                     value_function):
    pair_start_date = market_info.get_pair_start_time(pair)
    pair_end_date = market_info.get_pair_end_time(pair)

    actual_start_date = max(start_date, pair_start_date)
    actual_end_date = min(end_date, pair_end_date)

    market_info.set_market_date(actual_start_date)

    data_points = []

    while market_info.market_date <= actual_end_date:
        feature_set = compute_feature_set(market_info, pair, feature_functions)
        value = value_function(market_info, pair)
        data_points.append(DataPoint(feature_set=feature_set, value=value))
        market_info.inc_market_date()

    return DataSet(data_points=data_points)


def compute_feature_set(market_info, pair, feature_functions):
    features = []
    for func in feature_functions:
        new_features = func(market_info, pair)
        for feature in new_features:
            features.append(feature)
    return FeatureSet(features=features)
