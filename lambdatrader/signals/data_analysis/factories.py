from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME,
)
from lambdatrader.signals.data_analysis.df_datasets import DateRange, SplitDateRange
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVCloseDelta, OHLCVValue, OHLCVSelfDelta, DFFeatureSet, DummyFeature, RandomFeature,
)
from lambdatrader.signals.data_analysis.df_values import CloseReturn, MaxReturn
from lambdatrader.signals.data_analysis.utils import date_str_to_timestamp
from lambdatrader.utilities.utils import seconds


class FeaturesFactory:

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


ff = FeaturesFactory


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


class ValueSets:

    @classmethod
    def close_return_4h(cls):
        return DFFeatureSet(features=[Values.close_return_4h()])

    @classmethod
    def close_max_return_4h(cls):
        return DFFeatureSet(features=[Values.close_return_4h(), Values.max_return_4h()])

    @classmethod
    def close_return_next_candle(cls):
        return DFFeatureSet(features=[Values.close_return_next_candle()])

    @classmethod
    def close_max_return_next_candle(cls):
        return DFFeatureSet(features=[Values.close_return_next_candle(),
                                      Values.max_return_next_candle()])


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
    def january_20_days_test_20_days_val_160_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2018-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=160), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_360_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2018-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=360), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_500_days_train(cls):
        december_20_timestamp = date_str_to_timestamp('2018-12-20')
        return SplitDateRange(
            train_dr=DateRange(december_20_timestamp - seconds(days=500), december_20_timestamp),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )

    @classmethod
    def january_20_days_test_20_days_val_rest_train(cls):
        return SplitDateRange(
            train_dr=DateRange.from_str(None, '2018-12-20'),
            val_dr=DateRange.from_str('2017-12-20', '2018-01-10'),
            test_dr=DateRanges.january_last_20_days()
        )
