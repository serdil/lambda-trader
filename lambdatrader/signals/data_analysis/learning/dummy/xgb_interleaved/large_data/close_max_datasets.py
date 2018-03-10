from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.df_datasets import SplitDatasetDescriptor
from lambdatrader.signals.data_analysis.factories import SplitDateRanges, FeatureSets, ValueSets


class CloseMaxDatasets:

    def __init__(self, pairs='BTC_ETH', interleaved=False):
        if isinstance(pairs, str):
            pairs = [pairs]

        self.pairs = pairs
        self.interleaved = interleaved


    def get_1(self):
        split_date_range = SplitDateRanges.january_20_days_test_20_days_val_rest_train()
        feature_set = FeatureSets.get_all_periods_last_ten_ohlcv_now_delta()
        value_set_close = ValueSets.close_return_next_candle()
        value_set_max = ValueSets.max_return_next_candle()

        return self._create(split_date_range, feature_set, value_set_close, value_set_max)

    def get_2(self):
        split_date_range = SplitDateRanges.january_20_days_test_20_days_val_160_days_train()
        feature_set = FeatureSets.get_all_periods_last_ten_ohlcv_now_delta()
        value_set_close = ValueSets.close_return_next_candle()
        value_set_max = ValueSets.max_return_next_candle()

        return self._create(split_date_range, feature_set, value_set_close, value_set_max)

    def _create(self, split_date_range, feature_set, value_set_close, value_set_max):
        close_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=self.pairs, feature_set=feature_set, value_set=value_set_close,
            split_date_range=split_date_range, exchanges=(POLONIEX,), interleaved=self.interleaved)

        max_dataset = SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=self.pairs, feature_set=feature_set, value_set=value_set_max,
            split_date_range=split_date_range, exchanges=(POLONIEX,), interleaved=self.interleaved)

        return close_dataset, max_dataset
