import random
from operator import attrgetter

import numpy as np
import pandas
import pandas as pd
from typing import List

from functools import reduce

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


def remove_name(df):
    df.columns = [0]
    return df


def to_df(series_or_df):
    if isinstance(series_or_df, pd.Series):
        return series_or_df.to_frame(name=0)
    else:
        return series_or_df


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

    def compute(self, pair_dfs, selected_pair):
        raise NotImplementedError


class LookbackFeature(BaseFeature):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def lookback(self):
        raise NotImplementedError

    @property
    def period(self):
        raise NotImplementedError

    def compute(self, pair_dfs, selected_pair, normalize=True):
        dfs = pair_dfs[selected_pair]
        raw_df = self.compute_raw(dfs)
        if normalize:
            return to_ffilled_df_with_name(dfs[M5].index, raw_df, self.name)
        else:
            return raw_df

    def compute_raw(self, dfs):
        raise NotImplementedError

    @staticmethod
    def assert_periods_equal(*features):
        expected_period = features[0].period
        if not all((f.period is expected_period for f in features)):
            raise ArgumentError


class DummyFeature(LookbackFeature):

    def __init__(self, period=M5):
        self._period = period

    @property
    def name(self):
        return 'dummy_feature'

    @property
    def lookback(self):
        return 0

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        zero_series = pd.Series(np.zeros(len(dfs[M5])), index=dfs[M5].index)
        return to_df(zero_series)


class RandomFeature(LookbackFeature):

    def __init__(self, period=M5):
        self._period = period

    @property
    def name(self):
        return 'random_feature'

    @property
    def lookback(self):
        return 0

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        random_series = pd.Series(np.random.rand(len(dfs[M5])), index=dfs[M5].index)
        return to_df(random_series)


class OHLCVValue(LookbackFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self._period = period

    @property
    def name(self):
        return 'ohlcv_value_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                self.mode,
                                                                self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        return to_df(dfs[self.period][self.mode])


class ShiftedCloseValue(OHLCVValue):
    def __init__(self, offset=0, period=M5):
        super().__init__(mode=OHLCV_CLOSE, offset=offset, period=period)


class CloseValue(ShiftedCloseValue):
    def __init__(self, period=M5):
        super().__init__(offset=0, period=period)


class Shifted(LookbackFeature):
    def __init__(self, feature: LookbackFeature, offset=0, period=None):
        self.feature = feature
        self.offset = offset

    @property
    def name(self):
        return 'shifted_offset_{}_feature_({})'.format(self.offset, self.feature.name)

    @property
    def lookback(self):
        return self.feature.lookback + self.offset * self.period.seconds()

    @property
    def period(self):
        return self.feature.period

    def compute_raw(self, dfs):
        return to_df(self.feature.compute_raw(dfs).shift(self.offset))


class BinaryOpFeature(LookbackFeature):
    def __init__(self, f1, f2, period=None):
        self.assert_periods_equal(f1, f2)

        self.f1 = f1
        self.f2 = f2

    @property
    def name(self):
        return 'binary_op_{}_f1_({})_f2_({})'.format(self.op_name, self.f1.name, self.f2.name)

    @property
    def op_name(self):
        raise NotImplementedError

    @property
    def period(self):
        return self.f1.period

    @property
    def lookback(self):
        return max(self.f1.lookback, self.f2.lookback)

    def compute_raw(self, dfs):
        raise NotImplementedError


class Sum(BinaryOpFeature):
    @property
    def op_name(self):
        return 'sum'

    def compute_raw(self, dfs):
        return to_df(remove_name(self.f1.compute_raw(dfs)) + remove_name(self.f2.compute_raw(dfs)))


class Diff(BinaryOpFeature):
    @property
    def op_name(self):
        return 'diff'

    def compute_raw(self, dfs):
        return to_df(remove_name(self.f1.compute_raw(dfs)) - remove_name(self.f2.compute_raw(dfs)))


class Div(BinaryOpFeature):
    @property
    def op_name(self):
        return 'div'

    def compute_raw(self, dfs):
        return to_df(remove_name(self.f1.compute_raw(dfs)) / remove_name(self.f2.compute_raw(dfs)))


class Mult(BinaryOpFeature):
    @property
    def op_name(self):
        return 'mult'

    def compute_raw(self, dfs):
        return to_df(remove_name(self.f1.compute_raw(dfs)) * remove_name(self.f2.compute_raw(dfs)))


class FeatureFeature(LookbackFeature):
    def __init__(self, feature, period=None):
        self.feature = feature

    @property
    def name(self):
        return self.feature.name

    @property
    def lookback(self):
        return self.feature.lookback

    @property
    def period(self):
        return self.feature.period

    def compute_raw(self, dfs):
        return to_df(self.feature.compute_raw(dfs))


class NormDiff(FeatureFeature):
    def __init__(self, f1, f2, period=None):
        self.assert_periods_equal(f1, f2)

        diff = Diff(f1, f2)
        diff_div = Div(diff, f1)
        super().__init__(diff_div)


class ShiftedNormDiff(NormDiff):
    def __init__(self, f1, f2, offset, period=None):
        self.assert_periods_equal(f1, f2)

        shifted_f2 = Shifted(f2, offset=offset)
        super().__init__(f1, shifted_f2)


class ShiftedSelfNormDiff(ShiftedNormDiff):
    def __init__(self, feature, offset, period=None):
        super().__init__(feature, feature, offset=offset)


class ConstMult(FeatureFeature):
    def __init__(self, feature, constant, period=None):
        const_mult = Mult(feature, Constant(constant, period=feature.period))
        super().__init__(const_mult)


class UnaryOpFeature(LookbackFeature):
    def __init__(self, feature, period=None):
        self.feature = feature

    @property
    def name(self):
        return 'unary_op_{}_feature_({})'.format(self.op_name, self.feature.name)

    @property
    def op_name(self):
        raise NotImplementedError

    @property
    def lookback(self):
        return self.feature.lookback

    @property
    def period(self):
        return self.feature.period

    def compute_raw(self, dfs):
        raise NotImplementedError


class Sin(UnaryOpFeature):
    @property
    def op_name(self):
        return 'sin'

    def compute_raw(self, dfs):
        return to_df(np.sin(self.feature.compute_raw(dfs)))


class Cos(UnaryOpFeature):
    @property
    def op_name(self):
        return 'cos'

    def compute_raw(self, dfs):
        return to_df(np.cos(self.feature.compute_raw(dfs)))


class Square(UnaryOpFeature):
    @property
    def op_name(self):
        return 'square'

    def compute_raw(self, dfs):
        df = self.feature.compute_raw(dfs)
        return to_df(df * df)


class Cube(UnaryOpFeature):
    @property
    def op_name(self):
        return 'square'

    def compute_raw(self, dfs):
        df = self.feature.compute_raw(dfs)
        return to_df(df * df * df)


class Constant(LookbackFeature):
    def __init__(self, constant, period=M5):
        self.constant = constant
        self._period = period

    @property
    def name(self):
        return 'constant_({})'.format(self.constant)

    @property
    def lookback(self):
        return 0

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        return to_df(pandas.Series(self.constant, index=df.index))


class OHLCVNowSelfDelta(LookbackFeature):
    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self._period = period

    @property
    def name(self):
        return 'ohlcv_now_self_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                         self.mode,
                                                                         self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        self_delta = df[self.mode].diff(self.offset) / df[self.mode]
        return to_df(self_delta)


class OHLCVNowCloseDelta(LookbackFeature):

    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self._period = period

    @property
    def name(self):
        return 'ohlcv_now_close_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                          self.mode,
                                                                          self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        close_delta = (df[OHLCV_CLOSE] - df[self.mode].shift(self.offset)) / df[OHLCV_CLOSE]
        return to_df(close_delta)


class OHLCVSelfCloseDelta(LookbackFeature):

    def __init__(self, mode, offset, period=M5):
        self.mode = mode
        self.offset = offset
        self._period = period

    @property
    def name(self):
        return 'ohlcv_self_close_delta_period_{}_mode_{}_offset_{}'.format(self.period.name,
                                                                           self.mode,
                                                                           self.offset)

    @property
    def lookback(self):
        return self.period.seconds() * self.offset

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        close = df[OHLCV_CLOSE]
        shifted_close = close.shift(self.offset)
        shifted_mode = df[self.mode].shift(self.offset)
        self_close_delta = (shifted_close - shifted_mode) / shifted_close
        return to_df(self_close_delta)


class IndicatorValue(LookbackFeature):

    def __init__(self, indicator, args, offset, longest_timeperiod, period=M5, output_col=None):
        self.indicator = indicator
        self.args = args
        self.offset = offset
        self._period = period
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

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        value = self.indicator.function()(df, *self.args).shift(self.offset)
        if self.output_col is not None:
            value = value.iloc[:,self.output_col:self.output_col+1]
        return to_df(value)

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
        self._period = period
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

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        ind_values = self.indicator.function()(df, *self.args)
        self_delta = (ind_values.diff(self.offset) / ind_values)
        if self.output_col is not None:
            self_delta = self_delta.iloc[:,self.output_col:self.output_col+1]
        return to_df(self_delta)

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
        self._period = period
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

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        close_delta = (self.indicator.function()(df, *self.args).shift(self.offset)
                       .rsub(df[OHLCV_CLOSE], axis=0).div(df[OHLCV_CLOSE], axis=0))
        if self.output_col is not None:
            close_delta = close_delta.iloc[:,self.output_col:self.output_col+1]
        return to_df(close_delta)

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
        self._period = period
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

    @property
    def period(self):
        return self._period

    def compute_raw(self, dfs):
        df = dfs[self.period]
        shifted_close = df[OHLCV_CLOSE].shift(self.offset)
        close_delta = (self.indicator.function()(df, *self.args).shift(self.offset)
                       .rsub(shifted_close, axis=0).div(shifted_close, axis=0))
        if self.output_col is not None:
            close_delta = close_delta.iloc[:,self.output_col:self.output_col+1]
        return to_df(close_delta)

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
    def __init__(self, features: List[LookbackFeature], coef: List[float], period=None):
        self.assert_periods_equal(*features)

        if len(features) != len(coef):
            raise ArgumentError
        self.features = features
        self.coef = coef

    @property
    def name(self):
        feature_names_str = join_list([f.name for f in self.features])
        coef_str = join_list(self.coef)
        return 'linear_comb_features_({})_coef_{}'.format(feature_names_str, coef_str)

    @property
    def lookback(self):
        return max([f.lookback for f in self.features])

    @property
    def period(self):
        return self.features[0].period

    def compute_raw(self, dfs):
        coef_dfs = [f.compute_raw(dfs) * coef for coef, f in zip(self.coef, self.features)]
        for df in coef_dfs:
            remove_name(df)
        sum_df = sum(coef_dfs)
        return to_df(sum_df)


class PolynomialFeatureCombination(LookbackFeature):
    def __init__(self, features: List[LookbackFeature], exp: List[float], period=None):
        self.assert_periods_equal(*features)

        if len(features) != len(exp):
            raise ArgumentError
        self.features = features
        self.exp = exp

    @property
    def name(self):
        feature_names_str = join_list([f.name for f in self.features])
        exp_str = join_list(self.exp)
        return 'polynomial_comb_features_({})_exp_{}'.format(feature_names_str, exp_str)

    @property
    def lookback(self):
        return max([f.lookback for f in self.features])

    @property
    def period(self):
        return self.features[0].period

    def compute_raw(self, dfs):
        exp_dfs = [f.compute_raw(dfs) ** coef for coef, f in zip(self.exp, self.features)]
        for df in exp_dfs:
            remove_name(df)
        mult_df = reduce(lambda a, b: a*b, exp_dfs)
        return to_df(mult_df)


class AbsValue(LookbackFeature):
    def __init__(self, feature: LookbackFeature, period=None):
        self.feature = feature

    @property
    def name(self):
        return 'abs_value_feature_({})'.format(self.feature.name)

    @property
    def lookback(self):
        return self.feature.lookback

    @property
    def period(self):
        return self.feature.period

    def compute_raw(self, dfs):
        df_abs = self.feature.compute_raw(dfs).abs()
        return to_df(df_abs)


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


class IndicatorValueOnFeature(LookbackFeature):

    def __init__(self, feature,
                 indicator, args, offset, longest_timeperiod, output_col=None, period=None):
        self.feature = feature

        self.indicator = indicator
        self.args = args
        self.offset = offset
        self.longest_timeperiod = longest_timeperiod
        self.output_col = output_col

    @property
    def name(self):
        return ('indicator_value_on_feature_'
                'feature_({})_indicator_{}_args_{}_offset_{}_output_{}'
                .format(self.feature.name, self.indicator.name,
                        join_list(self.args), self.offset, self.output_col))

    @property
    def lookback(self):
        return self.feature.lookback + \
               (self.longest_timeperiod + self.offset) * self.feature.period.seconds()

    @property
    def period(self):
        return self.feature.period

    def compute_raw(self, dfs):
        ind_input = self.feature.compute_raw(dfs)
        value = self.indicator.function()(ind_input, *self.args).shift(self.offset)
        if self.output_col is not None:
            value = value.iloc[:,self.output_col:self.output_col+1]
        return to_df(value)

    def assert_output_col_max(self, max_n_outputs):
        if (self.output_col is not None and
                (not isinstance(self.output_col, int)
                 or self.output_col >= max_n_outputs
                 or self.output_col < 0)):
            raise ArgumentError('Invalid output_col')


# TODO: add indicator features on features
