from lambdatrader.signals.data_analysis.features import (
    CANDLE_PERIODS, make_ohcl_delta, LOOKBACK_NUM_CANDLES, make_volume, fib_seq, make_sma_delta,
    make_rsi, make_atr,
)


def get_large_feature_func_set():
    for candle_period in CANDLE_PERIODS:
        yield make_ohcl_delta(LOOKBACK_NUM_CANDLES, candle_period)
        yield make_volume(LOOKBACK_NUM_CANDLES, candle_period)

        for period in fib_seq():
            yield make_sma_delta(LOOKBACK_NUM_CANDLES, candle_period, period)
            yield make_rsi(LOOKBACK_NUM_CANDLES, candle_period, period)
            yield make_atr(LOOKBACK_NUM_CANDLES, candle_period, period)
