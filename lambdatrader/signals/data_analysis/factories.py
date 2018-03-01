from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME,
)
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVCloseDelta, OHLCVValue, OHLCVSelfDelta, DFFeatureSet, DummyFeature, RandomFeature,
)


class DFFeaturesFactory:

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


ff = DFFeaturesFactory


class DFFeatureSetFactory:

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
