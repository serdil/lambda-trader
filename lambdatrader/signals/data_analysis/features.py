from typing import List

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import PeriodEnum
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.datasets import Feature
from lambdatrader.signals.data_analysis.utils import join_list
from lambdatrader.signals.utils import get_candle, get_indicator


def make_ohcl_delta(num_candles, candle_period: PeriodEnum):
    def feature_ohcl_delta(market_info: BacktestingMarketInfo, pair):
        current_close = get_candle(market_info, pair, 0, period=candle_period).close
        for i in range(num_candles):
            candle = get_candle(market_info, pair, i, period=candle_period)
            yield Feature(name='open_delta_period_{}_offset_{:03d}'.format(candle_period.name, i),
                          value=(current_close - candle.open) / current_close)
            yield Feature(name='high_delta_period_{}_offset_{:03d}'.format(candle_period.name, i),
                          value=(current_close - candle.high) / current_close)
            yield Feature(name='close_delta_period_{}_offset_{:03d}'.format(candle_period.name, i),
                          value=(current_close - candle.close) / current_close)
            yield Feature(name='low_delta_period_{}_offset_{:03d}'.format(candle_period.name, i),
                          value=(current_close - candle.low) / current_close)

    return feature_ohcl_delta


def make_volume(num_candles, candle_period):
    def feature_volume(market_info: BacktestingMarketInfo, pair):
        for i in range(num_candles):
            candle = get_candle(market_info, pair, i, period=candle_period)
            yield Feature(name='volume_period_{}_offset_{:03d}'.format(candle_period.name, i),
                          value=candle.base_volume)
    return feature_volume


def make_rsi(num_candles, candle_period: PeriodEnum, rsi_period):
    return make_indicator(num_candles, candle_period, IndicatorEnum.RSI, [rsi_period])


def make_atr(num_candles, candle_period: PeriodEnum, atr_period):
    return make_indicator(num_candles, candle_period, IndicatorEnum.ATR, [atr_period])


def make_indicator(num_candles, candle_period: PeriodEnum, indicator: IndicatorEnum, args: List):
    def feature_indicator(market_info: BacktestingMarketInfo, pair):
        for i in range(num_candles):
            indicator_values = get_indicator(market_info, pair, indicator, args, i, candle_period)
            for val_ind, value in enumerate(indicator_values):
                yield Feature(name='indicator_{}_args_{}_out_{}_period_{}_offset_{:03d}'
                              .format(indicator.name, join_list(args),
                                      val_ind, candle_period.name, i),
                              value=value)
    return feature_indicator


def make_sma_delta(num_candles, candle_period, sma_period):
    return make_indicator_delta(num_candles, candle_period, IndicatorEnum.SMA, [sma_period])


def make_ema_delta(num_candles, candle_period: PeriodEnum, ema_period):
    return make_indicator_delta(num_candles, candle_period, IndicatorEnum.EMA, [ema_period])


def make_indicator_delta(num_candles, candle_period: PeriodEnum,
                         indicator: IndicatorEnum, args: List):
    def feature_indicator_delta(market_info: BacktestingMarketInfo, pair):
        current_vals = list(get_indicator(market_info, pair, indicator, args, 0, candle_period))
        for i in range(num_candles):
            indicator_values = get_indicator(market_info, pair, indicator, args, i, candle_period)
            for val_ind, value in enumerate(indicator_values):
                yield Feature(name='indicator_delta_{}_args_{}_out_{}_period_{}_offset_{:03d}'
                              .format(indicator.name, join_list(args),
                                      val_ind, candle_period.name, i),
                              value=(current_vals[val_ind] - value) / current_vals[val_ind])
    return feature_indicator_delta

