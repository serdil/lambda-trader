from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import PeriodEnum
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.datasets import Feature
from lambdatrader.signals.utils import get_candle, get_indicator


def make_ohclv_delta(num_candles, candle_period: PeriodEnum):
    def _ohclv_delta(market_info: BacktestingMarketInfo, pair):
        current_close = get_candle(market_info, pair, 0, period=candle_period).close
        for i in range(num_candles):
            candle = get_candle(market_info, pair, i, period=candle_period)
            yield Feature(name='open_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=current_close - candle.open)
            yield Feature(name='high_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=current_close - candle.high)
            yield Feature(name='close_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=current_close - candle.close)
            yield Feature(name='low_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=current_close - candle.low)
            yield Feature(name='volume_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=current_close - candle.volume)

    return _ohclv_delta


def make_sma_delta(num_candles, sma_period, candle_period: PeriodEnum):
    def _sma_delta(market_info: BacktestingMarketInfo, pair):
        current_sma = get_indicator(market_info, pair,
                                    IndicatorEnum.SMA, [sma_period], 0, candle_period)
        for i in range(num_candles):
            sma = get_indicator(market_info, pair,
                                IndicatorEnum.SMA, [sma_period], i, candle_period)
            yield Feature(name='sma_delta_sma_period_{}_candle_period_{}_offset_{}'
                          .format(sma_period, candle_period.name, i),
                          value=current_sma - sma)
    return _sma_delta
