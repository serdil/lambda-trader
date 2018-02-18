from operator import attrgetter

from lambdatrader.constants import M5
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.constants import OHLCV_CLOSE
from lambdatrader.signals.data_analysis.utils import join_list


class DFFeatureSet:

    def __init__(self, features):
        self.features = []
        self.features.extend(features)
        self._sort_features()

    def add_features(self, features):
        self.features.extend(features)
        self._sort_features()

    def _sort_features(self):
        self.features.sort(key=attrgetter('name'))


class BaseFeature:

    @property
    def name(self):
        raise NotImplementedError

    def compute(self, df):
        raise NotImplementedError


class OHLCVValue(BaseFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_value_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                self.mode,
                                                                self.offset)

    def compute(self, df):
        return df[self.mode].shift(self.offset)


class OHLCVSelfDelta(BaseFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_self_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                     self.mode,
                                                                     self.offset)

    def compute(self, df):
        return df[self.mode].diff(self.offset)


class OHLCVCloseDelta(BaseFeature):

    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_close_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                      self.mode,
                                                                      self.offset)

    def compute(self, df):
        return df[OHLCV_CLOSE] - df[self.mode].shift(self.offset)


class IndicatorValue(BaseFeature):

    def __init__(self, indicator, args, offset, period=M5):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return ('indicator_value_period_{}_indicator_{}_args_{}_offset_{}'
                .format(self.period.name, self.indicator.name, join_list(self.args), self.offset))

    def compute(self, df):
        return self.indicator.function()(df, *self.args).shift(self.offset)


class IndicatorSelfDelta(BaseFeature):

    def __init__(self, indicator, args, offset, period=M5):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return ('indicator_self_delta_period_{}_indicator_{}_args_{}_offset_{}'
                .format(self.period.name, self.indicator.name, join_list(self.args), self.offset))

    def compute(self, df):
        self.indicator.function()(df, *self.args).diff(self.offset)


class IndicatorCloseDelta(BaseFeature):

    def __init__(self, indicator, args, offset, period=M5):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return ('indicator_close_delta_period_{}_indicator_{}_args_{}_offset_{}'
                .format(self.period.name, self.indicator.name, join_list(self.args), self.offset))

    def compute(self, df):
        return self.indicator.function()(df, *self.args).shift(self.offset)\
            .rsub(df[OHLCV_CLOSE], axis=0)


class MACDValue(IndicatorValue):
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9, offset=0, period=M5):
        super().__init__(IndicatorEnum.MACD, [fastperiod, slowperiod, signalperiod], offset, period)


class RSIValue(IndicatorValue):
    def __init__(self, timeperiod=14, offset=0, period=M5):
        super().__init__(IndicatorEnum.RSI, [timeperiod], offset, period)


class BBandsCloseDelta(IndicatorCloseDelta):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0, offset=0, period=M5):
        super().__init__(IndicatorEnum.BBANDS, [timeperiod, nbdevup, nbdevdn, matype],
                         offset, period)
