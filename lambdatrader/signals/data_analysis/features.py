from typing import List

from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.constants import PeriodEnum, M5, M15, H, H4, D
from lambdatrader.indicator_functions import IndicatorEnum
from lambdatrader.signals.data_analysis.datasets import Feature
from lambdatrader.signals.utils import get_candle, get_indicator

LOOKBACK_NUM_CANDLES = 15


def get_feature_funcs_iter():
    for candle_period in [M5, M15, H, H4, D]:
        yield make_ohcl_delta(LOOKBACK_NUM_CANDLES, candle_period)
        yield make_volume(LOOKBACK_NUM_CANDLES, candle_period)

        for period in fib_seq():
            yield make_sma_delta(LOOKBACK_NUM_CANDLES, candle_period, period)
            yield make_rsi(LOOKBACK_NUM_CANDLES, candle_period, period)
            yield make_atr(LOOKBACK_NUM_CANDLES, candle_period, period)


def fib_seq():
    yield 3
    yield 5
    yield 8
    yield 13
    yield 21
    yield 34


def make_ohcl_delta(num_candles, candle_period: PeriodEnum):
    def feature_ohcl_delta(market_info: BacktestingMarketInfo, pair):
        current_close = get_candle(market_info, pair, 0, period=candle_period).close
        for i in range(num_candles):
            candle = get_candle(market_info, pair, i, period=candle_period)
            yield Feature(name='open_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=(current_close - candle.open) / current_close)
            yield Feature(name='high_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=(current_close - candle.high) / current_close)
            yield Feature(name='close_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=(current_close - candle.close) / current_close)
            yield Feature(name='low_delta_period_{}_offset_{}'.format(candle_period.name, i),
                          value=(current_close - candle.low) / current_close)

    return feature_ohcl_delta


def make_volume(num_candles, candle_period):
    def feature_volume(market_info: BacktestingMarketInfo, pair):
        for i in range(num_candles):
            candle = get_candle(market_info, pair, i, period=candle_period)
            yield Feature(name='volume_period_{}_offset_{}'.format(candle_period.name, i),
                          value=candle.volume)
    return feature_volume


def make_rsi(num_candles, candle_period: PeriodEnum, rsi_period):
    return make_indicator(num_candles, candle_period, IndicatorEnum.RSI, [rsi_period])


def make_atr(num_candles, candle_period: PeriodEnum, atr_period):
    return make_indicator(num_candles, candle_period, IndicatorEnum.RSI, [atr_period])


def make_indicator(num_candles, candle_period: PeriodEnum, indicator: IndicatorEnum, args: List):
    def feature_indicator(market_info: BacktestingMarketInfo, pair):
        for i in range(num_candles):
            indicator_values = get_indicator(market_info, pair, indicator, args, i, candle_period)
            for val_ind, value in enumerate(indicator_values):
                yield Feature(name='indicator_{}_args_{}_out_{}_period_{}_offset_{}'
                              .format(indicator.name, join_list(args),
                                      val_ind, candle_period.name, i),
                              value=value)
    return feature_indicator


def make_sma_delta(num_candles, candle_period, sma_period):
    return make_indicator_delta(num_candles, candle_period, IndicatorEnum.SMA, [sma_period])


def make_indicator_delta(num_candles, candle_period: PeriodEnum,
                         indicator: IndicatorEnum, args: List):
    def feature_indicator_delta(market_info: BacktestingMarketInfo, pair):
        current_vals = list(get_indicator(market_info, pair, indicator, args, 0, candle_period))
        for i in range(num_candles):
            indicator_values = get_indicator(market_info, pair, indicator, args, i, candle_period)
            for val_ind, value in enumerate(indicator_values):
                yield Feature(name='indicator_delta_{}_args_{}_out_{}_period_{}_offset_{}'
                              .format(indicator.name, join_list(args),
                                      val_ind, candle_period.name, i),
                              value=(current_vals[val_ind] - value) / current_vals[val_ind])
    return feature_indicator_delta


def join_list(lst):
    return ','.join([str(elem) for elem in lst])
