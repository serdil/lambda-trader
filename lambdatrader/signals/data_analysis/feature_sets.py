from lambdatrader.constants import M5, M15, H, H4, D
from lambdatrader.signals.data_analysis.features import (
    make_ohcl_delta, make_volume, make_sma_delta, make_rsi, make_atr, make_ema_delta,
)


LARGE_SET_LOOKBACK_NUM_CANDLES = 15
LARGE_SET_CANDLE_PERIODS = [M5, M15, H, H4, D]


def get_small_feature_func_set():
    lookback = 5
    for candle_period in [M5, M15, H, H4]:
        yield make_ohcl_delta(lookback, candle_period)
        yield make_volume(lookback, candle_period)


def get_small_feature_func_set_with_indicators():
    lookback = 5
    for candle_period in [M5, M15, H, H4]:
        yield make_ohcl_delta(lookback, candle_period)
        yield make_volume(lookback, candle_period)

        yield make_ema_delta(1, candle_period, 8)
        yield make_ema_delta(1, candle_period, 13)
        yield make_ema_delta(1, candle_period, 21)
        yield make_ema_delta(1, candle_period, 55)

        yield make_atr(1, candle_period, 8)

        yield make_rsi(1, candle_period, 14)


def get_alt_small_feature_func_set():
    yield make_ohcl_delta(3, M5)
    yield make_volume(3, M5)

    yield make_ohcl_delta(4, M15)
    yield make_volume(4, M15)

    yield make_ohcl_delta(6, H4)
    yield make_volume(6, H4)

    yield make_ohcl_delta(1, D)
    yield make_volume(1, D)


def get_alt_small_feature_func_set_2():
    yield make_ohcl_delta(288 * 3, M5)


def get_large_feature_func_set():
    for candle_period in LARGE_SET_CANDLE_PERIODS:
        yield make_ohcl_delta(LARGE_SET_LOOKBACK_NUM_CANDLES, candle_period)
        yield make_volume(LARGE_SET_LOOKBACK_NUM_CANDLES, candle_period)

        for period in fib_seq():
            yield make_sma_delta(LARGE_SET_LOOKBACK_NUM_CANDLES, candle_period, period)
            yield make_rsi(LARGE_SET_LOOKBACK_NUM_CANDLES, candle_period, period)
            yield make_atr(LARGE_SET_LOOKBACK_NUM_CANDLES, candle_period, period)


def fib_seq():
    yield 3
    yield 5
    yield 8
    yield 13
    yield 21
    yield 34

