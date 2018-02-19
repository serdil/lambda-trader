from lambdatrader.constants import M5
from lambdatrader.signals.data_analysis.constants import OHLCV_HIGH, OHLCV_CLOSE
from lambdatrader.signals.data_analysis.df_features import BaseFeature, to_ffilled_df_with_name


class ValueFeature(BaseFeature):

    @property
    def name(self):
        raise NotImplementedError

    def compute(self, dfs):
        raise NotImplementedError


class MaxReturn(ValueFeature):

    def __init__(self, n_candles, period=M5):
        self.n_candles = n_candles
        self.period = period

    @property
    def name(self):
        return 'max_price_period_{}_n_candles_{}'.format(self.period.name, self.n_candles)

    def compute(self, dfs):
        max_returns = (dfs[self.period][OHLCV_HIGH].rolling(window=self.n_candles)
                       .max().shift(-self.n_candles) / dfs[self.period][OHLCV_CLOSE])
        return to_ffilled_df_with_name(dfs[M5].index, max_returns, self.name)
