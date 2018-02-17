from operator import attrgetter

from lambdatrader.constants import M5
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
        pass


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
        pass


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
        pass


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
        pass


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
        pass


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
        pass
