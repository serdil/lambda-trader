import random
from operator import attrgetter

import numpy as np
import pandas as pd
from typing import List

from lambdatrader.constants import M5
from lambdatrader.exceptions import ArgumentError
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.constants import (
    OHLCV_CLOSE, OHLCV_LOW, OHLCV_HIGH, OHLCV_OPEN,
)
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
        return max(0, 0,
                   *[f.lookback for f in self.features if isinstance(f, LookbackFeature)])

    def get_lookforward(self):
        from lambdatrader.signals.data_analysis.df_values import LookforwardFeature
        return max(0, 0,
                   *[f.lookforward for f in self.features if isinstance(f, LookforwardFeature)])

    @property
    def feature_names(self):
        return [f.name for f in self.features]

    @property
    def num_features(self):
        return len(self.features)

    def shrink(self, ratio):
        size = int(len(self.features) * ratio)
        return self.shrink_to_size(size)

    def shrink_to_size(self, size):
        return DFFeatureSet(features=random.sample(self.features, size), sort=self.sort)

    def sample(self, size=None):
        if size is None:
            size = len(self.features)
        return random.sample(self.features, size)


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


class OHLCVNowSelfDelta(LookbackFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_now_self_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                         self.mode,
                                                                         self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    def compute(self, dfs):
        df = dfs[self.period]
        self_delta = df[self.mode].diff(self.offset) / df[self.mode]
        return to_ffilled_df_with_name(dfs[M5].index, self_delta, self.name)


class OHLCVNowCloseDelta(LookbackFeature):

    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_now_close_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                          self.mode,
                                                                          self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    def compute(self, dfs):
        df = dfs[self.period]
        close_delta = (df[OHLCV_CLOSE] - df[self.mode].shift(self.offset)) / df[OHLCV_CLOSE]
        return to_ffilled_df_with_name(dfs[M5].index, close_delta, self.name)


class OHLCVSelfCloseDelta(LookbackFeature):

    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self.period = period

    @property
    def name(self):
        return 'ohlcv_self_close_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                           self.mode,
                                                                           self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    def compute(self, dfs):
        df = dfs[self.period]
        close = df[OHLCV_CLOSE]
        shifted_close = close.shift(self.offset)
        shifted_mode = df[self.mode].shift(self.offset)
        self_close_delta = (shifted_close - shifted_mode) / shifted_close
        return to_ffilled_df_with_name(dfs[M5].index, self_close_delta, self.name)


class IndicatorValue(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5, output_col=None):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod
        self.output_col = output_col

    @property
    def name(self):
        return ('indicator_value_period_{}_indicator_{}_args_{}_offset_{}_output_{}'
                .format(self.period.name, self.indicator.name,
                        join_list(self.args), self.offset, self.output_col))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        value = self.indicator.function()(df, *self.args).shift(self.offset)
        if self.output_col is not None:
            value = value.iloc[:,self.output_col:self.output_col+1]
        return to_ffilled_df_with_name(dfs[M5].index,
                                       value,
                                       self.name)

    def assert_output_col_max(self, max_n_outputs):
        if (self.output_col is not None and
                (not isinstance(self.output_col, int)
                 or self.output_col >= max_n_outputs
                 or self.output_col < 0)):
            raise ArgumentError('Invalid output_col')


class IndicatorSelfDelta(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5, output_col=None):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod
        self.output_col = output_col

    @property
    def name(self):
        return ('indicator_self_delta_period_{}_indicator_{}_args_{}_offset_{}_output_{}'
                .format(self.period.name, self.indicator.name,
                        join_list(self.args), self.offset, self.output_col))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        ind_values = self.indicator.function()(df, *self.args)
        self_delta = (ind_values.diff(self.offset) / ind_values)
        if self.output_col is not None:
            self_delta = self_delta.iloc[:,self.output_col:self.output_col+1]
        return to_ffilled_df_with_name(dfs[M5].index, self_delta, self.name)

    def assert_output_col_max(self, max_n_outputs):
        if (self.output_col is not None and
                (not isinstance(self.output_col, int)
                 or self.output_col >= max_n_outputs
                 or self.output_col < 0)):
            raise ArgumentError('Invalid output_col')


class IndicatorNowCloseDelta(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5, output_col=None):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod
        self.output_col = output_col

    @property
    def name(self):
        return ('indicator_now_close_delta_period_{}_indicator_{}_args_{}_offset_{}_output_{}'
                .format(self.period.name, self.indicator.name,
                        join_list(self.args), self.offset, self.output_col))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        close_delta = (self.indicator.function()(df, *self.args).shift(self.offset)
                       .rsub(df[OHLCV_CLOSE], axis=0).div(df[OHLCV_CLOSE], axis=0))
        if self.output_col is not None:
            close_delta = close_delta.iloc[:,self.output_col:self.output_col+1]
        return to_ffilled_df_with_name(dfs[M5].index, close_delta, self.name)

    def assert_output_col_max(self, max_n_outputs):
        if (self.output_col is not None and
                (not isinstance(self.output_col, int)
                 or self.output_col >= max_n_outputs
                 or self.output_col < 0)):
            raise ArgumentError('Invalid output_col')


class IndicatorSelfCloseDelta(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5, output_col=None):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.period = period
        self.longest_timeperiod = longest_timeperiod
        self.output_col = output_col

    @property
    def name(self):
        return ('indicator_self_close_delta_period_{}_indicator_{}_args_{}_offset_{}_output_{}'
                .format(self.period.name, self.indicator.name,
                        join_list(self.args), self.offset, self.output_col))

    @property
    def lookback(self):
        return (self.longest_timeperiod + self.offset) * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        shifted_close = df[OHLCV_CLOSE].shift(self.offset)
        close_delta = (self.indicator.function()(df, *self.args).shift(self.offset)
                       .rsub(shifted_close, axis=0).div(shifted_close, axis=0))
        if self.output_col is not None:
            close_delta = close_delta.iloc[:,self.output_col:self.output_col+1]
        return to_ffilled_df_with_name(dfs[M5].index, close_delta, self.name)

    def assert_output_col_max(self, max_n_outputs):
        if (self.output_col is not None and
                (not isinstance(self.output_col, int)
                 or self.output_col >= max_n_outputs
                 or self.output_col < 0)):
            raise ArgumentError('Invalid output_col')


class MACDValue(IndicatorValue):
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9, offset=0, period=M5,
                 output_col=None):
        longest_timeperiod = max(fastperiod, slowperiod, signalperiod)
        super().__init__(IndicatorEnum.MACD,
                         [fastperiod, slowperiod, signalperiod],
                         offset, longest_timeperiod, period, output_col)
        self.assert_output_col_max(3)


class RSIValue(IndicatorValue):
    def __init__(self, timeperiod=14, offset=0, period=M5):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.RSI, [timeperiod], offset, longest_timeperiod, period)


class BBandsNowCloseDelta(IndicatorNowCloseDelta):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0, offset=0, period=M5,
                 output_col=None):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.BBANDS, [timeperiod, nbdevup, nbdevdn, matype],
                         offset, longest_timeperiod, period, output_col)
        self.assert_output_col_max(3)


class SymmetricBBandsNowCloseDelta(BBandsNowCloseDelta):
    def __init__(self, timeperiod=5, nbdev=2, matype=0, offset=0, period=M5,
                 output_col=None):
        super().__init__(timeperiod=timeperiod,
                         nbdevup=nbdev,
                         nbdevdn=nbdev,
                         matype=matype,
                         offset=offset,
                         period=period,
                         output_col=output_col)


class BBandsSelfCloseDelta(IndicatorSelfCloseDelta):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0, offset=0, period=M5,
                 output_col=None):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.BBANDS, [timeperiod, nbdevup, nbdevdn, matype],
                         offset, longest_timeperiod, period, output_col)
        self.assert_output_col_max(3)


class SymmetricBBandsSelfCloseDelta(BBandsSelfCloseDelta):
    def __init__(self, timeperiod=5, nbdev=2, matype=0, offset=0, period=M5,
                 output_col=None):
        super().__init__(timeperiod=timeperiod,
                         nbdevup=nbdev,
                         nbdevdn=nbdev,
                         matype=matype,
                         offset=offset,
                         period=period,
                         output_col=output_col)


class CandlestickPattern(IndicatorValue):
    def __init__(self, pattern_indicator: IndicatorEnum, offset=0, period=M5):
        longest_timeperiod = 25
        super().__init__(pattern_indicator, [], offset, longest_timeperiod, period)


class SMASelfCloseDelta(IndicatorSelfCloseDelta):
    def __init__(self, timeperiod=30, offset=0, period=M5):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.SMA, [timeperiod], offset, longest_timeperiod, period)


class SMANowCloseDelta(IndicatorNowCloseDelta):
    def __init__(self, timeperiod=30, offset=0, period=M5):
        longest_timeperiod = timeperiod
        super().__init__(IndicatorEnum.SMA, [timeperiod], offset, longest_timeperiod, period)


class LinearFeatureCombination(LookbackFeature):
    def __init__(self, features: List[LookbackFeature], coef: List[float]):
        if len(features) != len(coef):
            raise ArgumentError
        self.features = features
        self.coef = coef

    @property
    def name(self):
        feature_names_str = join_list([f.name for f in self.features])
        coef_str = join_list(self.coef)
        return 'linear_comb_features_{}_coef_{}'.format(feature_names_str, coef_str)

    @property
    def lookback(self):
        return max([f.lookback for f in self.features])

    def compute(self, dfs):
        coef_dfs = [f.compute(dfs) * coef for coef, f in zip(self.coef, self.features)]
        for df in coef_dfs:
            df.columns = list(range(df.shape[1]))
        sum_df = sum(coef_dfs)
        return to_ffilled_df_with_name(dfs[M5].index, sum_df, self.name)


class AbsValue(LookbackFeature):
    def __init__(self, feature: LookbackFeature):
        self.feature = feature

    @property
    def name(self):
        return 'abs_value_feature_{}'.format(self.feature.name)

    @property
    def lookback(self):
        return self.feature.lookback

    def compute(self, dfs):
        df_abs = self.feature.compute(dfs).abs()
        return to_ffilled_df_with_name(dfs[M5].index, df_abs, self.name)


class BBandsBandWidth(LinearFeatureCombination):
    def __init__(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0, offset=0, period=M5):
        lower_band_delta = BBandsSelfCloseDelta(timeperiod=timeperiod,
                                                nbdevup=nbdevup,
                                                nbdevdn=nbdevdn,
                                                matype=matype,
                                                offset=offset,
                                                period=period,
                                                output_col=2)
        upper_band_delta = BBandsSelfCloseDelta(timeperiod=timeperiod,
                                                nbdevup=nbdevup,
                                                nbdevdn=nbdevdn,
                                                matype=matype,
                                                offset=offset,
                                                period=period,
                                                output_col=0)
        super().__init__(features=[lower_band_delta, upper_band_delta],
                         coef=[1, -1])


class CandlestickTipToTipSize(LinearFeatureCombination):
    def __init__(self, offset=0, period=M5):
        low_delta = OHLCVSelfCloseDelta(mode=OHLCV_LOW, offset=offset, period=period)
        high_delta = OHLCVSelfCloseDelta(mode=OHLCV_HIGH, offset=offset, period=period)
        super().__init__(features=[low_delta, high_delta],
                         coef=[1, -1])


class CandlestickBodySize(AbsValue):
    def __init__(self, offset=0, period=M5):
        close_delta = OHLCVSelfCloseDelta(mode=OHLCV_CLOSE, offset=offset, period=period)
        open_delta = OHLCVSelfCloseDelta(mode=OHLCV_OPEN, offset=offset, period=period)
        close_minus_open = LinearFeatureCombination(features=[close_delta, open_delta],
                                                    coef=[1, -1])
        super().__init__(feature=close_minus_open)
