from lambdatrader.constants import M5
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME,
)
from lambdatrader.signals.data_analysis.df_features import (
    OHLCVCloseDelta, OHLCVValue, OHLCVSelfDelta, DFFeatureSet,
)


class DFFeatureSetFactory:

    def get_feature_set_1(self):
        features = []
        features.extend(DFFeaturesFactory.get_ohlc_close_delta())
        features.extend(DFFeaturesFactory.get_volume_value())
        features.extend(DFFeaturesFactory.get_volume_self_delta())
        return DFFeatureSet(features=features)


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
