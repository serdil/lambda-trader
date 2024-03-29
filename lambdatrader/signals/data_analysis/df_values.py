import numpy as np
import pandas as pd

from lambdatrader.constants import M5
from lambdatrader.signals.data_analysis.constants import OHLCV_HIGH, OHLCV_CLOSE, OHLCV_LOW
from lambdatrader.signals.data_analysis.df_features import BaseFeature, to_ffilled_df_with_name


class LookforwardFeature(BaseFeature):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def lookforward(self):
        raise NotImplementedError

    def compute(self, dfs):
        raise NotImplementedError


class ValueFeature(LookforwardFeature):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def lookforward(self):
        raise NotImplementedError

    def compute(self, dfs):
        raise NotImplementedError


class DummyValue(ValueFeature):
    @property
    def name(self):
        return 'dummy_value'

    @property
    def lookforward(self):
        return 0

    def compute(self, dfs):
        zero_series = pd.Series(np.zeros(len(dfs[M5])), index=dfs[M5].index)
        return to_ffilled_df_with_name(dfs[M5].index, zero_series, self.name)


class MaxReturn(ValueFeature):

    def __init__(self, n_candles, period=M5):
        self.n_candles = n_candles
        self.period = period

    @property
    def name(self):
        return 'max_return_period_{}_n_candles_{}'.format(self.period.name, self.n_candles)

    @property
    def lookforward(self):
        return self.n_candles * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        max_returns = (df[OHLCV_HIGH].rolling(window=self.n_candles).max().shift(-self.n_candles) /
                       df[OHLCV_CLOSE]) - 1
        return to_ffilled_df_with_name(dfs[M5].index, max_returns, self.name)


class CloseReturn(ValueFeature):

    def __init__(self, n_candles, period=M5):
        self.n_candles = n_candles
        self.period = period

    @property
    def name(self):
        return 'close_return_period_{}_n_candles_{}'.format(self.period.name, self.n_candles)

    @property
    def lookforward(self):
        return self.n_candles * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        close_returns = (df[OHLCV_CLOSE].diff(self.n_candles).shift(-self.n_candles) /
                         df[OHLCV_CLOSE])
        return to_ffilled_df_with_name(dfs[M5].index, close_returns, self.name)


class MinReturn(ValueFeature):

    def __init__(self, n_candles, period=M5):
        self.n_candles = n_candles
        self.period = period

    @property
    def name(self):
        return 'min_return_period_{}_n_candles_{}'.format(self.period.name, self.n_candles)

    @property
    def lookforward(self):
        return self.n_candles * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        min_returns = (df[OHLCV_LOW].rolling(window=self.n_candles).min().shift(-self.n_candles) /
                       df[OHLCV_CLOSE]) - 1
        return to_ffilled_df_with_name(dfs[M5].index, min_returns, self.name)


class CloseAvgReturn(ValueFeature):

    def __init__(self, n_candles, period=M5):
        self.n_candles = n_candles
        self.period = period

    @property
    def name(self):
        return 'close_avg_return_period_{}_n_candles_{}'.format(self.period.name, self.n_candles)

    @property
    def lookforward(self):
        return self.n_candles * self.period.seconds()

    def compute(self, dfs):
        df = dfs[self.period]
        avg_returns = (df[OHLCV_CLOSE].rolling(window=self.n_candles).mean()
                       .shift(-self.n_candles) / df[OHLCV_CLOSE]) - 1
        return to_ffilled_df_with_name(dfs[M5].index, avg_returns, self.name)
