from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.exchanges.enums import POLONIEX
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME,
)
from lambdatrader.signals.data_analysis.df_datasets import (
    DateRange, SplitDateRange, SplitDatasetDescriptor,
)
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVCloseDelta, OHLCVValue, OHLCVSelfDelta, DFFeatureSet, DummyFeature, RandomFeature,
)
from lambdatrader.signals.data_analysis.df_values import CloseReturn, MaxReturn, DummyValue
from lambdatrader.signals.data_analysis.utils import date_str_to_timestamp
from lambdatrader.utilities.utils import seconds


class FeatureListFactory:

    @classmethod
    def get_ohlc_close_delta(cls, offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        features = []
        for period in periods:
            for offset in offsets:
                features.append(OHLCVCloseDelta(OHLCV_OPEN, offset=offset, period=period))
                features.append(OHLCVCloseDelta(OHLCV_HIGH, offset=offset, period=period))
                features.append(OHLCVCloseDelta(OHLCV_LOW, offset=offset, period=period))
                features.append(OHLCVCloseDelta(OHLCV_CLOSE, offset=offset, period=period))
        return features

    @classmethod
    def get_volume_value(cls, offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        features = []
        for period in periods:
            for offset in offsets:
                features.append(OHLCVValue(OHLCV_VOLUME, offset=offset, period=period))
        return features

    @classmethod
    def get_ohlc_close_delta_volume_value(cls, offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        ohlc_close_delta = cls.get_ohlc_close_delta(offsets=offsets, periods=periods)
        volume_value = cls.get_volume_value(offsets=offsets, periods=periods)
        return ohlc_close_delta + volume_value

    @classmethod
    def get_volume_self_delta(cls, offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        features = []
        for period in periods:
            for offset in offsets:
                features.append(OHLCVSelfDelta(OHLCV_VOLUME, offset=offset, period=period))
        return features


ff = FeatureListFactory


class FeatureSets:

    @classmethod
    def get_feature_set_1(cls):
        features = []
        features.extend(ff.get_ohlc_close_delta_volume_value())
        features.extend(ff.get_volume_self_delta())
        return DFFeatureSet(features=features)

    @classmethod
    def get_feature_set_2(cls):
        features = []
        periods = [M5, M15, H, H4, D]
        features.extend(ff.get_ohlc_close_delta_volume_value(periods=periods))
        features.extend(ff.get_volume_self_delta(periods=periods))
        return DFFeatureSet(features=features)

    @classmethod
    def get_ohlcv_all_periods_with_num_offsets(cls, num_offsets=(5, 5, 5, 5, 5,)):
        features = []
        periods = [M5, M15, H, H4, D]
        if len(num_offsets) != len(periods):
            raise ValueError('number of num_offsets should be {}'.format(len(periods)))
        for i, period in enumerate(periods):
            features.extend(ff.get_ohlc_close_delta_volume_value(periods=[period],
                                                                 offsets=range(num_offsets[i])))
        return DFFeatureSet(features=features)

    @classmethod
    def get_smallest(cls):
        features = ff.get_ohlc_close_delta_volume_value(offsets=[1], periods=[M5])
        return DFFeatureSet(features=features)

    @classmethod
    def get_small(cls):
        num_offsets = 5
        periods = [M5, M15, H, H4]
        return cls._get_ohlc_close_delta_volume_value_num_offsets_periods(num_offsets=num_offsets,
                                                                          periods=periods)

    @classmethod
    def get_all_periods_last_five_ohlcv(cls):
        num_offsets = 5
        periods = [M5, M15, H, H4, D]
        return cls._get_ohlc_close_delta_volume_value_num_offsets_periods(num_offsets=num_offsets,
                                                                          periods=periods)

    @classmethod
    def get_all_periods_last_ten_ohlcv(cls):
        num_offsets = 10
        periods = [M5, M15, H, H4, D]
        return cls._get_ohlc_close_delta_volume_value_num_offsets_periods(num_offsets=num_offsets,
                                                                          periods=periods)

    @classmethod
    def get_all_periods_last_n_ohlcv(cls, n):
        num_offsets = n
        periods = [M5, M15, H, H4, D]
        return cls._get_ohlc_close_delta_volume_value_num_offsets_periods(num_offsets=num_offsets,
                                                                          periods=periods)

    @classmethod
    def get_dummy(cls):
        return DFFeatureSet(features=[DummyFeature()])

    @classmethod
    def get_random(cls):
        return DFFeatureSet(features=[RandomFeature()])

    @classmethod
    def _get_ohlc_close_delta_volume_value_num_offsets_periods(cls, num_offsets, periods):
        features = ff.get_ohlc_close_delta_volume_value(offsets=range(num_offsets), periods=periods)
        return DFFeatureSet(features=features)


class Values:

    @classmethod
    def close_return_4h(cls):
        return cls.close_return_n_candles(48)

    @classmethod
    def close_return_1h(cls):
        return cls.close_return_n_candles(12)

    @classmethod
    def close_return_next_candle(cls):
        return cls.close_return_n_candles(1)

    @classmethod
    def close_return_n_candles(cls, n):
        return CloseReturn(n_candles=n)

    @classmethod
    def max_return_4h(cls):
        return cls.max_return_n_candles(48)

    @classmethod
    def max_return_1h(cls):
        return cls.max_return_n_candles(12)

    @classmethod
    def max_return_next_candle(cls):
        return cls.max_return_n_candles(1)

    @classmethod
    def max_return_n_candles(cls, n):
        return MaxReturn(n_candles=n)

    @classmethod
    def dummy(cls):
        return DummyValue()


class ValueSets:

    @classmethod
    def close_return_4h(cls):
        return DFFeatureSet(features=[Values.close_return_4h()])

    @classmethod
    def max_return_4h(cls):
        return DFFeatureSet(features=[Values.max_return_4h()])

    @classmethod
    def close_max_return_4h(cls):
        return DFFeatureSet(features=[Values.close_return_4h(), Values.max_return_4h()])

    @classmethod
    def close_return_next_candle(cls):
        return DFFeatureSet(features=[Values.close_return_next_candle()])

    @classmethod
    def max_return_next_candle(cls):
        return DFFeatureSet(features=[Values.max_return_next_candle()])

    @classmethod
    def close_max_return_next_candle(cls):
        return DFFeatureSet(features=[Values.close_return_next_candle(),
                                      Values.max_return_next_candle()])

    @classmethod
    def dummy(cls):
        return DFFeatureSet(features=[Values.dummy()])


class Dates:
    @classmethod
    def january_10(cls):
        return date_str_to_timestamp('2018-01-10')

    @classmethod
    def feb_1(cls):
        return date_str_to_timestamp('2018-02-01')

    @classmethod
    def n_days_before_january_10(cls, n):
        return cls.january_10() - seconds(days=n)

    @classmethod
    def sixty_days_before_january_10(cls):
        return cls.n_days_before_january_10(60)

    @classmethod
    def n_days_before_date(cls, date, n):
        return date - seconds(days=n)


class DateRanges:

    @classmethod
    def december(cls):
        return DateRange.from_str(start_str='2017-12-01', end_str='2018-01-01')

    @classmethod
    def january(cls):
        return DateRange.from_str(start_str='2018-01-01', end_str='2018-02-01')

    @classmethod
    def february(cls):
        return DateRange.from_str(start_str='2018-02-01', end_str='2018-03-01')

    @classmethod
    def until_start_of_january(cls):
        return DateRange.from_str(start_str=None, end_str='2018-01-01')

    @classmethod
    def until_start_of_february(cls):
        return DateRange.from_str(start_str=None, end_str='2018-02-01')

    @classmethod
    def january_last_20_days(cls):
        return DateRange.from_str(start_str='2018-01-10', end_str='2018-02-01')

    @classmethod
    def january_last_10_days(cls):
        return DateRange.from_str(start_str='2018-01-20', end_str='2018-02-01')

    @classmethod
    def sixty_days_until_january_10(cls):
        return DateRange(Dates.sixty_days_before_january_10(), Dates.january_10())

    @classmethod
    def until_start_of_january_5_months(cls):
        return DateRange.from_str(start_str='2017-07-01', end_str='2018-01-01')


class SplitDateRanges:

    @classmethod
    def january_1_month_test_1_month_val_5_months_train(cls):
        return SplitDateRange(
            train_dr=DateRanges.until_start_of_january_5_months(),
            val_dr=DateRanges.december(),
            test_dr=DateRanges.january()
        )

    @classmethod
    def january_20_days_test_20_days_val_20_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2017-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=20), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_160_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2017-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=160), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_360_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2017-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=360), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_500_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2017-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=500), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_rest_train(cls):
        return SplitDateRange(
            train_dr=DateRange.from_str(None, '2017-12-20'),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_60_days_val_rest_train(cls):
        return SplitDateRange(
            train_dr=DateRange.from_str(None, '2017-11-10'),
            val_dr=DateRange.from_str('2017-11-10', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_3_days_test_3_days_val_7_days_train(cls):
        return SplitDateRange(
            train_dr=DateRange.from_str('2018-01-17', '2018-01-24'),
            val_dr=DateRange.from_str('2018-01-24', '2018-01-27'),
            test_dr=DateRange.from_str('2018-01-27', '2018-01-30')
        )

    @classmethod
    def date_n_days_test_m_days_val_k_days_train(cls, date, test_days, val_days, train_days):
        test_start = Dates.n_days_before_date(date, test_days)
        val_start = Dates.n_days_before_date(date, test_days + val_days)
        train_start = Dates.n_days_before_date(date, test_days + val_days + train_days)
        return SplitDateRange(
            train_dr=DateRange(train_start, val_start),
            val_dr=DateRange(val_start, test_start),
            test_dr=DateRange(test_start, date)
        )

    @classmethod
    def jan_n_days_test_m_days_val_k_days_train(cls, test_days, v, t):
        return cls.date_n_days_test_m_days_val_k_days_train(Dates.feb_1(),
                                                            test_days, v, t)


class SplitDatasetDescriptors:

    @classmethod
    def sdd_1(cls):
        return SplitDatasetDescriptor.create_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_max_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_160_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_close(cls):
        return SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_160_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_max(cls):
        return SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.max_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_160_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_close_mini(cls):
        return SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_return_4h(),
            split_date_range=SplitDateRanges.january_3_days_test_3_days_val_7_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_max_mini(cls):
        return SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.max_return_4h(),
            split_date_range=SplitDateRanges.january_3_days_test_3_days_val_7_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_more_data(cls):
        return SplitDatasetDescriptor.create_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_max_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_360_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_more_data_close(cls):
        return SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_360_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True)

    @classmethod
    def sdd_1_more_data_max(cls):
        return SplitDatasetDescriptor.create_single_value_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.max_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_360_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_1_all_data(cls):
        return SplitDatasetDescriptor.create_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_max_return_4h(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_rest_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_2(cls):
        return SplitDatasetDescriptor.create_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_max_return_next_candle(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_160_days_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )

    @classmethod
    def sdd_2_all_data(cls):
        return SplitDatasetDescriptor.create_with_train_val_test_date_ranges(
            pairs=None,
            feature_set=FeatureSets.get_all_periods_last_ten_ohlcv(),
            value_set=ValueSets.close_max_return_next_candle(),
            split_date_range=SplitDateRanges.january_20_days_test_20_days_val_rest_train(),
            exchanges=(POLONIEX,),
            interleaved=True
        )
