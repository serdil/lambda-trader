from operator import attrgetter

import numpy as np
import pandas as pd

from lambdatrader.constants import M5
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.constants import OHLCV_CLOSE
from lambdatrader.signals.data_analysis.utils import join_list


def to_ffilled_df_with_name(index, series_or_df, name):
    if isinstance(series_or_df, pd.Series):
        df = series_or_df.to_frame(name=name)
    else:
        df = prefix_df_col_names(series_or_df, prefix=name)
    return df.reindex(index=index).fillna(method='ffill')


def prefix_df_col_names(df, prefix):
    df.rename(columns=lambda name: '{}:{}'.format(prefix, name), inplace=True)
    return df


class DFFeatureSet:

    def __init__(self, features, sort=True):
        self.features = []
        self.features.extend(features)
        self.sort = sort
        self._sort_features()

    def add_features(self, features):
        self.features.extend(features)
        self._sort_features()

    def _sort_features(self):
        if self.sort:
            self.features.sort(key=attrgetter('name'))

    def get_lookback(self):
        return max(0, *[f.lookback for f in self.features if isinstance(f, LookbackFeature)])

    def get_lookforward(self):
        from lambdatrader.signals.data_analysis.df_values import LookforwardFeature
        return max(0, *[f.lookforward for f in self.features if isinstance(f, LookforwardFeature)])


class BaseFeature:

    @property
    def name(self):
        raise NotImplementedError

    def compute(self, dfs):
        raise NotImplementedError


class LookbackFeature(BaseFeature):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def lookback(self):
        raise NotImplementedError

    def compute(self, dfs):
        raise NotImplementedError


class DummyFeature(LookbackFeature):

    @property
    def lookback(self):
        return 0

    @property
    def name(self):
        return 'dummy_feature'

    def compute(self, dfs):
        zero_series = pd.Series(np.zeros(len(dfs[M5])), index=dfs[M5].index)
        return to_ffilled_df_with_name(dfs[M5].index, zero_series, self.name)


class RandomFeature(LookbackFeature):

    @property
    def name(self):
        return 'random_feature'

    @property
    def lookback(self):
        return 0

    def compute(self, dfs):
        random_series = pd.Series(np.random.rand(len(dfs[M5])), index=dfs[M5].index)
        return to_ffilled_df_with_name(dfs[M5].index, random_series, self.name)


class OHLCVValue(LookbackFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_value_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                self.mode,
                                                                self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index, df[self.mode].shift(self.offset), self.name)


class OHLCVSelfDelta(LookbackFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_self_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                     self.mode,
                                                                     self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    def compute(self, dfs):
        df = dfs[self.period]
        self_delta = df[self.mode].diff(self.offset) / df[self.mode]
        return to_ffilled_df_with_name(dfs[M5].index, self_delta, self.name)


class OHLCVCloseDelta(LookbackFeature):

    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_close_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                      self.mode,
                                                                      self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    def compute(self, dfs):
        df = dfs[self.period]
        close_delta = (df[OHLCV_CLOSE] - df[self.mode].shift(self.offset)) / df[OHLCV_CLOSE]
        return to_ffilled_df_with_name(dfs[M5].index, close_delta, self.name)


class IndicatorValue(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod

    @property
    def name(self):
        return ('indicator_value_period_{}_indicator_{}_args_{}_offset_{}'
                .format(self.period.name, self.indicator.name, join_list(self.args), self.offset))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index,
                                       self.indicator.function()(df, *self.args).shift(self.offset),
                                       self.name)


class IndicatorSelfDelta(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod

    @property
    def name(self):
        return ('indicator_self_delta_period_{}_indicator_{}_args_{}_offset_{}'
                .format(self.period.name, self.indicator.name, join_list(self.args), self.offset))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        ind_values = self.indicator.function()(df, *self.args)
        self_delta = (ind_values.diff(self.offset) / ind_values)
        return to_ffilled_df_with_name(dfs[M5].index, self_delta, self.name)


class IndicatorCloseDelta(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod

    @property
    def name(self):
        return ('indicator_close_delta_period_{}_indicator_{}_args_{}_offset_{}'
                .format(self.period.name, self.indicator.name, join_list(self.args), self.offset))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        close_delta = (self.indicator.function()(df, *self.args).shift(self.offset)
                       .rsub(df[OHLCV_CLOSE], axis=0).div(df[OHLCV_CLOSE], axis=0))
        return to_ffilled_df_with_name(dfs[M5].index, close_delta, self.name)


class MACDValue(IndicatorValue):
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9, offset=0, period=M5):
        longest_timeperiod = max(fastperiod, slowperiod, signalperiod)
        super().__init__(IndicatorEnum.MACD,
                         [fastperiod, slowperiod, signalperiod],
                         offset, longest_timeperiod, period)


class RSIValue(IndicatorValue):
    def __init__(self, timeperiod=14, offset=0, period=M5):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.RSI, [timeperiod], offset, longest_timeperiod, period)


class BBandsCloseDelta(IndicatorCloseDelta):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0, offset=0, period=M5):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.BBANDS, [timeperiod, nbdevup, nbdevdn, matype],
                         offset, longest_timeperiod, period)
