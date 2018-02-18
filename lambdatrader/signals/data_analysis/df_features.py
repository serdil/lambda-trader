from operator import attrgetter

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

    def compute(self, dfs):
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

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index, df[self.mode].shift(self.offset), self.name)


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

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index, df[self.mode].diff(self.offset), self.name)


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

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index,
                                       df[OHLCV_CLOSE] - df[self.mode].shift(self.offset), self.name)


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

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index,
                                       self.indicator.function()(df, *self.args).shift(self.offset),
                                       self.name)


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

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index,
                                       self.indicator.function()(df, *self.args).diff(self.offset),
                                       self.name)


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

    def compute(self, dfs):
        df = dfs[self.period]
        return to_ffilled_df_with_name(dfs[M5].index,
                                       self.indicator.function()(df, *self.args).shift(self.offset)
                                       .rsub(df[OHLCV_CLOSE], axis=0), self.name)


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
