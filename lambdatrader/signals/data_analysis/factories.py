from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME,
)
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVCloseDelta, OHLCVValue, OHLCVSelfDelta, DFFeatureSet,
)


class DFFeaturesFactory:

    @staticmethod
    def get_ohlc_close_delta(offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        features = []
        for period in periods:
            for offset in offsets:
                features.append(OHLCVCloseDelta(OHLCV_OPEN, offset=offset, period=period))
                features.append(OHLCVCloseDelta(OHLCV_HIGH, offset=offset, period=period))
                features.append(OHLCVCloseDelta(OHLCV_LOW, offset=offset, period=period))
                features.append(OHLCVCloseDelta(OHLCV_CLOSE, offset=offset, period=period))
        return features

    @staticmethod
    def get_volume_value(offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        features = []
        for period in periods:
            for offset in offsets:
                features.append(OHLCVValue(OHLCV_VOLUME, offset=offset, period=period))
        return features

    @staticmethod
    def get_volume_self_delta(offsets=(1, 2, 3, 4, 5,), periods=(M5,)):
        features = []
        for period in periods:
            for offset in offsets:
                features.append(OHLCVSelfDelta(OHLCV_VOLUME, offset=offset, period=period))
        return features


ff = DFFeaturesFactory


class DFFeatureSetFactory:

    @staticmethod
    def get_feature_set_1():
        features = []
        features.extend(ff.get_ohlc_close_delta())
        features.extend(ff.get_volume_value())
        features.extend(ff.get_volume_self_delta())
        return DFFeatureSet(features=features)

    @staticmethod
    def get_feature_set_2():
        features = []
        periods = [M5, M15, H, H4, D]
        features.extend(ff.get_ohlc_close_delta(periods=periods))
        features.extend(ff.get_volume_value(periods=periods))
        features.extend(ff.get_volume_self_delta(periods=periods))
        return DFFeatureSet(features=features)

    @staticmethod
    def get_ohlcv_all_periods_with_num_offsets(offsets=(5,5,5,5,5,)):
        features = []
        periods = [M5, M15, H, H4, D]
        if len(offsets) != len(periods):
            raise ValueError('number of num_offsets should be {}'.format(len(periods)))
        for i, period in enumerate(periods):
            features.extend(ff.get_ohlc_close_delta(periods=[period], offsets=range(offsets[i])))
            features.extend(ff.get_volume_value(periods=[period], offsets=range(offsets[i])))
        return DFFeatureSet(features=features)
